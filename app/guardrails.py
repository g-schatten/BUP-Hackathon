"""Deterministic guardrails over untrusted LLM directive output.

Problem Statement Section 08: "LLM output must be treated as untrusted
structured data until deterministic validation passes." This module never
trusts anything the model produced without checking it, and it never invents
a new directive type. Where a value is out of range but repairable (e.g. an
hour list with a stray duplicate, a reserve slightly above capacity), it
clamps/repairs rather than discarding the whole directive to no_op - a
repaired directive still earns downstream-application credit; a discarded
one earns nothing (Participant Guide Section 09: penalty language treats a
relevant note marked no_op as a lost-credit case, not a safe default).

Only when a directive cannot be made sense of at all (missing required
numeric field, no valid hours) does it fall back to no_op for that note -
this is the SAFE FAILURE path: never crash, never invent a rule.
"""
from __future__ import annotations

ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

REQUIRED_HOURS_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
}


def _clean_hours(raw) -> list[int] | None:
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    try:
        ints = [int(h) for h in raw]
    except (TypeError, ValueError):
        return None
    ints = sorted({h for h in ints if 0 <= h <= 23})
    return ints if ints else None


def _no_op(note_index: int, reason: str) -> dict:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _repair_one(candidate: dict, battery_capacity: float) -> dict:
    note_index = candidate.get("note_index")
    directive_type = candidate.get("directive_type")
    explanation = candidate.get("explanation")
    explanation = explanation if isinstance(explanation, str) and explanation.strip() else "Interpreted from operator note."

    if directive_type not in ALLOWED_TYPES:
        return _no_op(note_index, "Unsupported or unrecognized directive type; treated as no_op.")

    if directive_type == "no_op":
        return _no_op(note_index, explanation)

    adj = candidate.get("structured_adjustment")
    adj = adj if isinstance(adj, dict) else {}

    hours = _clean_hours(adj.get("hours")) if directive_type in REQUIRED_HOURS_TYPES else None
    if directive_type in REQUIRED_HOURS_TYPES and hours is None:
        return _no_op(note_index, "Directive had no valid hour range; treated as no_op.")

    if directive_type == "solar_reduction":
        factor = adj.get("factor")
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            return _no_op(note_index, "Solar reduction factor missing or invalid; treated as no_op.")
        factor = max(0.0, min(1.0, factor))
        adjustment = {"hours": hours, "factor": factor}

    elif directive_type == "minimum_battery_reserve":
        level = adj.get("minimum_energy_kwh")
        try:
            level = float(level)
        except (TypeError, ValueError):
            return _no_op(note_index, "Reserve level missing or invalid; treated as no_op.")
        level = max(0.0, min(level, battery_capacity))
        adjustment = {"hours": hours, "minimum_energy_kwh": level}

    elif directive_type == "no_charge_window":
        adjustment = {"hours": hours}

    elif directive_type == "no_discharge_window":
        adjustment = {"hours": hours}

    elif directive_type == "max_grid_window":
        cap = adj.get("max_grid_kwh")
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            return _no_op(note_index, "Grid cap missing or invalid; treated as no_op.")
        cap = max(0.0, cap)
        adjustment = {"hours": hours, "max_grid_kwh": cap}

    else:  # pragma: no cover - unreachable, ALLOWED_TYPES covers all cases
        return _no_op(note_index, "Unsupported directive type; treated as no_op.")

    return {
        "note_index": note_index,
        "applies": True,
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": explanation,
    }


def apply_guardrails(
    operator_notes: list[str], raw_directives: list[dict], battery_capacity: float
) -> list[dict]:
    """Validate/repair LLM output into exactly one entry per note, in order.

    `raw_directives` is untrusted and may be malformed, missing entries,
    duplicated, out of range, or an unsupported type. The result always has
    exactly `len(operator_notes)` entries, note_index 0..N-1, each satisfying
    the Section 08 guardrail table.
    """
    n = len(operator_notes)
    by_index: dict[int, dict] = {}

    if isinstance(raw_directives, list):
        for cand in raw_directives:
            if not isinstance(cand, dict):
                continue
            idx = cand.get("note_index")
            if not isinstance(idx, int) or idx < 0 or idx >= n:
                continue
            if idx in by_index:
                continue  # first mapping for a given note wins; drop duplicates
            by_index[idx] = _repair_one(cand, battery_capacity)

    result = []
    for i in range(n):
        if i in by_index:
            result.append(by_index[i])
        else:
            result.append(_no_op(i, "No interpretation was produced for this note; treated as no_op."))
    return result
