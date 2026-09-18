"""End-to-end harness against a LIVE HTTP endpoint (exercises the real LLM ->
guardrails -> optimizer pipeline, unlike tests/run_samples.py which bypasses
the LLM entirely).

Usage:
    python3 tests/test_endpoint.py [base_url]
    (default base_url: http://localhost:8000)

For each public sample case, this:
  1. POSTs the raw scenario (with its real operator_notes) to /optimize-energy.
  2. Compares the returned directive_interpretation against the sample pack's
     expected interpretation (type/applies/hours/numeric values, tolerance
     0.01) - explanation text is never compared (the spec says it isn't
     judged byte-for-byte).
  3. Replays the returned hourly_plan against the SERVER'S OWN returned
     directives using the same validator the service runs internally, plus
     checks reported totals match recalculated ones.
  4. Reports p50/p95 latency and a cost-quality ratio (min(1, ref/ours)) for
     cases where the interpretation was fully correct and the plan is valid.

This is a public-sample-only sanity check - it does not replace the hidden
judge harness, but it catches interpretation regressions and latency issues
before submission.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.validator import validate_plan

SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "problem-statement"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)
TOL = 0.01


def post_json(url: str, payload: dict, timeout: float = 35.0) -> tuple[int, dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.monotonic() - t0
            return resp.status, json.loads(resp.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        try:
            return e.code, json.loads(e.read()), elapsed
        except Exception:
            return e.code, {}, elapsed


def _hours_match(a, b) -> bool:
    return list(a or []) == list(b or [])


def compare_interpretation(expected: list[dict], actual: list[dict]) -> list[str]:
    problems = []
    actual_by_idx = {e.get("note_index"): e for e in actual}
    for exp in expected:
        idx = exp["note_index"]
        got = actual_by_idx.get(idx)
        if got is None:
            problems.append(f"note {idx}: missing from response")
            continue
        if got.get("applies") != exp["applies"]:
            problems.append(f"note {idx}: applies={got.get('applies')} expected={exp['applies']}")
        if got.get("directive_type") != exp["directive_type"]:
            problems.append(
                f"note {idx}: directive_type={got.get('directive_type')} expected={exp['directive_type']}"
            )
            continue
        exp_adj = exp.get("structured_adjustment")
        got_adj = got.get("structured_adjustment")
        if exp_adj is None:
            continue
        if not isinstance(got_adj, dict):
            problems.append(f"note {idx}: structured_adjustment missing/invalid")
            continue
        if not _hours_match(exp_adj.get("hours"), got_adj.get("hours")):
            problems.append(f"note {idx}: hours={got_adj.get('hours')} expected={exp_adj.get('hours')}")
        for numeric_key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if numeric_key in exp_adj:
                got_val = got_adj.get(numeric_key)
                if got_val is None or abs(float(got_val) - float(exp_adj[numeric_key])) > TOL:
                    problems.append(
                        f"note {idx}: {numeric_key}={got_val} expected={exp_adj[numeric_key]}"
                    )
    return problems


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    data = json.loads(SAMPLE_PATH.read_text())
    cases = data["cases"]

    latencies = []
    interp_correct = 0
    plan_valid = 0
    quality_ratios = []
    failures = 0

    for case in cases:
        inp = case["input"]
        expected = case["expected_output"]
        scenario_id = inp["scenario_id"]

        status, resp, elapsed = post_json(f"{base_url}/optimize-energy", inp)
        latencies.append(elapsed)

        if status != 200:
            print(f"[FAIL] {scenario_id}: HTTP {status} {resp}")
            failures += 1
            continue

        interp_problems = compare_interpretation(expected["directive_interpretation"], resp.get("directive_interpretation", []))
        interp_ok = not interp_problems
        if interp_ok:
            interp_correct += 1

        applied = [d for d in resp.get("directive_interpretation", []) if d.get("applies")]
        hours = inp["hours"]
        battery = inp["battery"]
        totals = (resp.get("total_grid_kwh"), resp.get("total_cost_bdt"), resp.get("peak_grid_kwh"))
        try:
            plan_errors = validate_plan(hours, battery, applied, resp.get("hourly_plan", []), reported_totals=totals)
        except Exception as e:  # malformed plan shape
            plan_errors = [f"validator crashed: {e}"]

        valid = not plan_errors
        if valid:
            plan_valid += 1

        ref_cost = expected["total_cost_bdt"]
        our_cost = resp.get("total_cost_bdt", float("inf"))
        if interp_ok and valid:
            ratio = 1.0 if ref_cost <= TOL else min(1.0, ref_cost / max(our_cost, TOL))
            quality_ratios.append(ratio)

        status_str = "PASS" if interp_ok and valid else "FAIL"
        if status_str == "FAIL":
            failures += 1
        print(
            f"[{status_str}] {scenario_id}  latency={elapsed:.2f}s  interp_ok={interp_ok}  "
            f"plan_valid={valid}  cost={our_cost} ref={ref_cost}"
        )
        for p in interp_problems:
            print(f"    interp! {p}")
        for e in plan_errors[:5]:
            print(f"    plan! {e}")

    n = len(cases)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else float("nan")
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else float("nan")
    avg_quality = sum(quality_ratios) / len(quality_ratios) if quality_ratios else 0.0

    print()
    print(f"interpretation correct: {interp_correct}/{n}")
    print(f"plan valid:             {plan_valid}/{n}")
    print(f"avg cost quality ratio (valid+correct cases): {avg_quality:.4f}")
    print(f"latency p50={p50:.2f}s p95={p95:.2f}s")
    print(f"{n - failures}/{n} cases fully passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
