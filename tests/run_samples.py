"""Offline harness: replay the public sample pack through OUR optimizer +
validator using the pack's own expected directive_interpretation (bypassing
the LLM entirely). This isolates "is the optimizer/validator correct" from
"can the LLM extract directives correctly" - the latter is tested separately
by hitting the live HTTP endpoint (see tests/test_endpoint.py).

Usage:
    python3 tests/run_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.optimizer import compute_totals, relax_and_solve
from app.validator import validate_plan

SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "problem-statement"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


def main() -> int:
    data = json.loads(SAMPLE_PATH.read_text())
    cases = data["cases"]
    failures = 0

    for case in cases:
        inp = case["input"]
        expected = case["expected_output"]
        scenario_id = inp["scenario_id"]

        directives = [
            d for d in expected["directive_interpretation"] if d["applies"]
        ]

        hours = [h for h in inp["hours"]]
        battery = inp["battery"]

        plan, dropped = relax_and_solve(hours, battery, directives)
        total_grid, total_cost, peak_grid = compute_totals(plan, hours)

        errors = validate_plan(
            hours, battery, directives, plan,
            reported_totals=(total_grid, total_cost, peak_grid),
        )

        ref_cost = expected["total_cost_bdt"]
        cost_diff = abs(total_cost - ref_cost)
        cost_ok = cost_diff <= 0.01

        status = "PASS" if not errors and cost_ok and not dropped else "FAIL"
        if status == "FAIL":
            failures += 1

        print(f"[{status}] {scenario_id}  cost={total_cost} ref={ref_cost} diff={cost_diff:.4f} dropped={dropped}")
        for e in errors:
            print(f"    ! {e}")

    print()
    print(f"{len(cases) - failures}/{len(cases)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
