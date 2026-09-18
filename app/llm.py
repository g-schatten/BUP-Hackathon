"""LLM-backed operator-note interpretation (Problem Statement Section 02, 08).

This is the ONLY place a language model touches the pipeline, and its output
is never trusted directly - `app.guardrails.apply_guardrails` re-validates
and repairs everything before it reaches the optimizer. This module's job is
just to turn natural-language notes into a best-effort structured guess.

Model: Claude Haiku 4.5 (`claude-haiku-4-5`) - chosen for the Performance &
Reliability p95<=5s latency requirement. See PLAN.md for the accuracy/latency
trade-off measurement against claude-sonnet-5; swap MODEL_ID below if that
measurement favors Sonnet.

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

MODEL_ID = os.environ.get("GRIDWISE_LLM_MODEL", "claude-haiku-4-5")
REQUEST_TIMEOUT_S = float(os.environ.get("GRIDWISE_LLM_TIMEOUT_S", "12.0"))

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

SYSTEM_PROMPT = """You are the operator-note interpreter for a campus energy scheduling system (GridWise).

You will receive a battery specification and a numbered list of 1-3 short operator notes about \
a 24-hour campus energy schedule (hours 0-23). Convert EACH note into exactly one structured \
directive. Some notes are realistic distractors (facilities/admin chatter unrelated to the \
energy schedule) - mark those no_op. Never invent a directive type outside the list below, and \
never invent numeric values the note does not support.

SUPPORTED DIRECTIVE TYPES (use exactly these strings for directive_type):
- solar_reduction: usable solar is reduced during specific hours. Needs hours + factor.
  factor = the FRACTION OF SOLAR THAT REMAINS (not the amount removed).
  Example: "drops to 20%" -> factor 0.2. "an 80% reduction" -> factor 0.2 (100%-80%=20% remains).
  "roughly half" -> factor 0.5. "one-fifth of normal" -> factor 0.2.
- minimum_battery_reserve: battery must stay at or above a level during specific hours.
  Needs hours + minimum_energy_kwh (an absolute kWh number).
  If the note gives a PERCENTAGE OF BATTERY CAPACITY (e.g. "50% of capacity"), you MUST compute
  minimum_energy_kwh = that percentage * the battery's capacity_kwh given below. Do not leave it
  as a percentage.
- no_charge_window: battery charging is unavailable during specific hours. Needs hours only.
- no_discharge_window: battery discharging is unavailable during specific hours. Needs hours only.
- max_grid_window: grid import may not exceed a stated kWh amount during specific hours.
  Needs hours + max_grid_kwh.
- no_op: the note does not affect today's 24-hour energy schedule (distractor, or unrelated
  campus/admin news). hours/factor/minimum_energy_kwh/max_grid_kwh must all be omitted (null).

HOUR CONVENTION (critical - get this exactly right):
- Hours are whole-hour integers 0-23. The 24-hour day is hour 0 = 12:00-1:00 AM ... hour 23 =
  11:00 PM-midnight.
- A window is START-INCLUSIVE, END-EXCLUSIVE. "1 PM to 3 PM" or "13:00 to 15:00" means the window
  covers hour 13 and hour 14, but NOT hour 15 -> hours = [13, 14].
- "6 PM until 9 PM" -> hours = [18, 19, 20] (not 21).
- "from noon until 2 PM" -> noon is hour 12 -> hours = [12, 13].
- A window that wraps past midnight, e.g. "10 PM until 2 AM", covers hours 22, 23, 0, 1 -> return
  them SORTED ASCENDING: hours = [0, 1, 22, 23].
- "all day" / "for the entire day" / "throughout the day" -> hours = [0,1,2,...,23] (all 24).
- hours must be unique integers, ascending, each between 0 and 23 inclusive.

GENERAL RULES:
- Produce exactly one entry per note, with note_index matching the note's position (starting
  at 0), in the same order the notes are given.
- For every non-no_op directive, applies must be true. no_op is the only directive with
  applies = false.
- Do not change or assume anything about demand, tariff, or battery parameters other than what
  a supported directive explicitly lets you change.
- explanation: one short sentence justifying the interpretation.

WORKED EXAMPLES (paraphrases of the same rule are common in real notes - match the underlying
rule, not the exact wording):
1. "Solar output will drop to about 20% from 1 PM to 3 PM."
   -> directive_type=solar_reduction, hours=[13,14], factor=0.2
2. "PV production will drop to about 20% between 13:00 and 15:00."
   -> same as example 1: directive_type=solar_reduction, hours=[13,14], factor=0.2
3. "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."
   -> directive_type=solar_reduction, hours=[13,14], factor=0.2 (80% reduction -> 20% remains)
4. "Panel washing from one until three will leave roughly one-fifth of normal solar output."
   -> directive_type=solar_reduction, hours=[13,14], factor=0.2
5. "Do not charge the battery between 2 PM and 4 PM."
   -> directive_type=no_charge_window, hours=[14,15]
6. "The charging circuit will be unavailable from 2 AM until 5 AM for electrical maintenance."
   -> directive_type=no_charge_window, hours=[2,3,4]
7. "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
   -> directive_type=minimum_battery_reserve, hours=[18,19,20], minimum_energy_kwh=120
8. "Keep at least 50% of the battery capacity stored in the battery from 6 PM until 9 PM."
   (if battery capacity_kwh=200) -> directive_type=minimum_battery_reserve, hours=[18,19,20],
   minimum_energy_kwh=100  (0.5 * 200)
9. "For protection testing, the battery must not discharge from 6 PM until 8 PM."
   -> directive_type=no_discharge_window, hours=[18,19]
10. "The evening transformer limit is 180 kWh of grid import from 7 PM until 9 PM."
    -> directive_type=max_grid_window, hours=[19,20], max_grid_kwh=180
11. "The battery charger will be offline from 10 PM until 2 AM."
    -> directive_type=no_charge_window, hours=[0,1,22,23]
12. "Do not charge the battery at any point today."
    -> directive_type=no_charge_window, hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
13. "The cafeteria menu changes tomorrow." / "The library is extending book-return hours next
    week." / "A seminar room booking was moved to next week."
    -> directive_type=no_op (all fields null except explanation)

Respond using the emit_directive_interpretation tool only."""


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


TOOL_SCHEMA = {
    "name": "emit_directive_interpretation",
    "description": "Return the structured directive interpretation for every operator note.",
    "strict": True,
    "input_schema": {
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
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed; falling back to no_op for all notes")
        return []

    user_content = json.dumps(
        {
            "battery": {"capacity_kwh": battery_capacity},
            "operator_notes": [
                {"note_index": i, "text": note} for i, note in enumerate(operator_notes)
            ],
        }
    )

    try:
        client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S, max_retries=1)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "emit_directive_interpretation"},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception:  # noqa: BLE001 - any provider failure is a safe-failure case
        logger.exception("LLM interpretation call failed; falling back to no_op for all notes")
        return []

    tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is None:
        logger.warning("LLM response had no tool_use block (stop_reason=%s)", response.stop_reason)
        return []

    try:
        parsed = RawInterpretation.model_validate(tool_use.input)
    except Exception:  # noqa: BLE001 - malformed model output is untrusted data
        logger.exception("LLM tool_use input failed schema validation")
        return []

    raw = _to_raw_directives(parsed)
    _cache[key] = raw
    return raw
