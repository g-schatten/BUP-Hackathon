"""LLM-backed operator-note interpretation (Problem Statement Section 02, 08).

This is the ONLY place a language model touches the pipeline, and its output
is never trusted directly - `app.guardrails.apply_guardrails` re-validates
and repairs everything before it reaches the optimizer. This module's job is
just to turn natural-language notes into a best-effort structured guess.

Model: Groq-hosted Llama 3.3 70B Versatile (`llama-3.3-70b-versatile`) - a
free-tier, tool-calling-capable model served on Groq's LPU inference, chosen
for the Performance & Reliability p95<=5s latency requirement (Groq serves
this model at ~280 tokens/sec, far faster than typical API latency). See
PLAN.md for the free-tier rate-limit notes; swap MODEL_ID below if a
different Groq model is preferred.

SAFE FAILURE (Section 08): if the provider errors, times out, or returns
something unusable, this module returns an empty list rather than raising -
`apply_guardrails` turns a missing mapping into a safe no_op for that note.
It never crashes the request and never invents a directive on its own.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger("gridwise.llm")

MODEL_ID = os.environ.get("GRIDWISE_LLM_MODEL", "openai/gpt-oss-20b")
REQUEST_TIMEOUT_S = float(os.environ.get("GRIDWISE_LLM_TIMEOUT_S", "12.0"))
# gpt-oss / qwen3 models on Groq are reasoning models; "medium" (their default)
# produces highly variable, sometimes very long reasoning traces before the
# final tool call - observed 12-20s wall-clock on some public sample cases,
# blowing the p95<=5s Performance & Reliability budget. "low" keeps latency
# consistent with a small, measured accuracy cost (see PLAN.md for the
# before/after comparison). Ignored for models that don't support the field.
REASONING_EFFORT = os.environ.get("GRIDWISE_LLM_REASONING_EFFORT", "low")
REASONING_CAPABLE_MARKERS = ("gpt-oss", "qwen")

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

SYSTEM_PROMPT = """Interpret campus operator notes for GridWise energy scheduling (24 hours, 0-23).
Convert EACH note to exactly one directive. Admin/facilities chatter unrelated to energy -> no_op.
Never invent a type outside this list or a number the note doesn't support.

Types (directive_type) and their required fields:
- solar_reduction: hours + factor (FRACTION REMAINING, not removed: "80% reduction"->0.2, "drops
  to 20%"->0.2, "half"->0.5, "one-fifth"->0.2).
- minimum_battery_reserve: hours + minimum_energy_kwh (absolute kWh). If given as % of capacity,
  compute minimum_energy_kwh = pct * capacity_kwh (capacity given below) - never leave as a %.
- no_charge_window: hours only (charging blocked).
- no_discharge_window: hours only (discharging blocked).
- max_grid_window: hours + max_grid_kwh (grid import cap).
- no_op: everything null (distractor / irrelevant note).

Hours: integers 0-23, unique, ascending. Windows are start-inclusive/end-exclusive: list every
hour from start up to but NOT including end - the count is (end-start) hours. "1-3 PM" or
"13:00-15:00" -> [13,14] (2 hours, not 15). "6 PM until 9 PM" -> [18,19,20] (3 hours). "6 PM
until 10 PM" -> [18,19,20,21] (4 hours - don't drop the last one). Noon = hour 12. A window
wrapping midnight, e.g. "10 PM until 2 AM" -> [0,1,22,23] (sorted). "all day" -> [0..23].

Rules: one entry per note, note_index = its position, same order given. applies=true for every
non-no_op type; only no_op uses applies=false. Don't alter demand/tariff/battery params yourself.
explanation: one short sentence.

Examples (paraphrases mean the same rule):
"Solar drops to ~20% from 1-3 PM" / "80% reduction in solar during the 1-3 PM window" ->
  solar_reduction hours=[13,14] factor=0.2
"Do not charge 2-4 PM" -> no_charge_window hours=[14,15]
"Keep >=120 kWh in reserve 6-9 PM" -> minimum_battery_reserve hours=[18,19,20] minimum_energy_kwh=120
"Keep >=50% of capacity in reserve 6-9 PM" (capacity=200) -> minimum_battery_reserve
  hours=[18,19,20] minimum_energy_kwh=100
"No discharge 6-8 PM" -> no_discharge_window hours=[18,19]
"Grid import capped at 180 kWh 7-9 PM" -> max_grid_window hours=[19,20] max_grid_kwh=180
"Charger offline 10 PM until 2 AM" -> no_charge_window hours=[0,1,22,23]
"The cafeteria menu changes tomorrow" -> no_op

Call the emit_directive_interpretation tool only. No plain text."""


class RawDirectiveEntry(BaseModel):
    note_index: int
    applies: bool
    directive_type: str
    hours: Optional[list[int]] = None
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None
    explanation: str = ""


class RawInterpretation(BaseModel):
    directive_interpretation: list[RawDirectiveEntry]


# OpenAI-compatible function-calling schema (Groq's chat.completions API mirrors
# the OpenAI tool-calling shape: {"type": "function", "function": {...}}).
# `strict: True` requires every property to be listed in `required` and every
# object to set `additionalProperties: False` - optional fields are modeled as
# `required` + nullable via `anyOf` with `null`, the standard strict-schema idiom.
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "emit_directive_interpretation",
        "description": "Return the structured directive interpretation for every operator note.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "directive_interpretation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "note_index": {"type": "integer"},
                            "applies": {"type": "boolean"},
                            "directive_type": {"type": "string", "enum": list(DIRECTIVE_TYPES)},
                            "hours": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "integer"}},
                                    {"type": "null"},
                                ]
                            },
                            "factor": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "minimum_energy_kwh": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "max_grid_kwh": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "explanation": {"type": "string"},
                        },
                        "required": [
                            "note_index",
                            "applies",
                            "directive_type",
                            "hours",
                            "factor",
                            "minimum_energy_kwh",
                            "max_grid_kwh",
                            "explanation",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["directive_interpretation"],
            "additionalProperties": False,
        },
    },
}

_cache: dict[str, list[dict]] = {}


def _cache_key(operator_notes: list[str], battery_capacity: float) -> str:
    payload = json.dumps({"notes": operator_notes, "capacity": battery_capacity}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_raw_directives(parsed: RawInterpretation) -> list[dict]:
    out = []
    for e in parsed.directive_interpretation:
        adj: dict = {}
        if e.hours is not None:
            adj["hours"] = e.hours
        if e.factor is not None:
            adj["factor"] = e.factor
        if e.minimum_energy_kwh is not None:
            adj["minimum_energy_kwh"] = e.minimum_energy_kwh
        if e.max_grid_kwh is not None:
            adj["max_grid_kwh"] = e.max_grid_kwh
        out.append(
            {
                "note_index": e.note_index,
                "applies": e.applies,
                "directive_type": e.directive_type,
                "structured_adjustment": adj if e.directive_type != "no_op" else None,
                "explanation": e.explanation,
            }
        )
    return out


def interpret_notes(operator_notes: list[str], battery_capacity: float) -> list[dict]:
    """Call the LLM once for all notes. Returns a list of raw (untrusted)
    directive dicts shaped for `app.guardrails.apply_guardrails`. Returns an
    empty list on any provider/parse failure - guardrails maps that to a
    safe no_op per note (never invents a directive, never crashes).
    """
    key = _cache_key(operator_notes, battery_capacity)
    if key in _cache:
        return _cache[key]

    try:
        import groq
    except ImportError:
        logger.error("groq package not installed; falling back to no_op for all notes")
        return []

    user_content = json.dumps(
        {
            "battery": {"capacity_kwh": battery_capacity},
            "operator_notes": [
                {"note_index": i, "text": note} for i, note in enumerate(operator_notes)
            ],
        }
    )

    kwargs: dict = {}
    if any(marker in MODEL_ID for marker in REASONING_CAPABLE_MARKERS) and REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT

    try:
        client = groq.Groq(timeout=REQUEST_TIMEOUT_S, max_retries=1)
        response = client.chat.completions.create(
            model=MODEL_ID,
            max_completion_tokens=1024,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=[TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": "emit_directive_interpretation"}},
            **kwargs,
        )
    except Exception:  # noqa: BLE001 - any provider failure is a safe-failure case
        logger.exception("LLM interpretation call failed; falling back to no_op for all notes")
        return []

    tool_calls = response.choices[0].message.tool_calls if response.choices else None
    if not tool_calls:
        logger.warning(
            "LLM response had no tool call (finish_reason=%s)",
            response.choices[0].finish_reason if response.choices else "unknown",
        )
        return []

    try:
        args = json.loads(tool_calls[0].function.arguments)
        parsed = RawInterpretation.model_validate(args)
    except Exception:  # noqa: BLE001 - malformed model output is untrusted data
        logger.exception("LLM tool call arguments failed JSON/schema validation")
        return []

    raw = _to_raw_directives(parsed)
    _cache[key] = raw
    return raw
