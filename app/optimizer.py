"""Linear-programming optimizer for the 24-hour GridWise schedule.

Model (Problem Statement Section 09):
  variables per hour h=0..23:
    grid[h]        >= 0                    grid electricity purchased
    solar_used[h]  in [0, eff_solar[h]]    solar actually used (curtailment allowed)
    chg[h]         in [0, max_charge[h]]   battery charge amount
    dis[h]         in [0, max_discharge[h]] battery discharge amount

  energy balance (every hour):
    grid[h] + solar_used[h] + dis[h] == demand[h] + chg[h]

  battery state:
    E[h] = initial_energy + sum_{k<=h} (chg[k] - dis[k])
    emin[h] <= E[h] <= capacity
    E[23] == initial_energy                (end-of-day neutrality)

  objective:
    minimize sum(grid[h] * tariff[h])

Directives modify this model deterministically (Section 05.3):
  solar_reduction          -> eff_solar[h] = solar[h] * factor for listed hours
  minimum_battery_reserve  -> emin[h] = max(base_min, directive_min) for listed hours
  no_charge_window         -> max_charge[h] = 0 for listed hours
  no_discharge_window      -> max_discharge[h] = 0 for listed hours
  max_grid_window          -> grid[h] upper bound = max_grid_kwh for listed hours
  no_op                    -> no change

This module is solver-only: it assumes directives have already passed the
deterministic guardrails in guardrails.py. It never trusts LLM output shape
directly - callers must pass validated dicts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

H = 24
EPS = 1e-9
ROUND_DP = 6


@dataclass
class ScheduleModel:
    """Per-hour effective limits after directives are applied."""

    demand: list[float]
    tariff: list[float]
    eff_solar: list[float]
    emin: list[float]  # active minimum battery reserve per hour
    max_charge: list[float]
    max_discharge: list[float]
    grid_cap: list[float | None]
    capacity: float
    initial_energy: float


def build_model(hours: list[dict], battery: dict, directives: list[dict]) -> ScheduleModel:
    """Apply a list of ALREADY-VALIDATED directive dicts to the base scenario.

    `directives` items look like:
        {"directive_type": "solar_reduction", "structured_adjustment": {"hours": [...], "factor": ...}}
    Only directives with a supported, non-no_op type are expected here; callers
    filter for `applies=True` before calling this function.
    """
    ordered = sorted(hours, key=lambda h: h["hour"])
    demand = [float(h["demand_kwh"]) for h in ordered]
    solar = [float(h["solar_kwh"]) for h in ordered]
    tariff = [float(h["tariff_bdt_per_kwh"]) for h in ordered]

    base_min = float(battery["minimum_energy_kwh"])
    capacity = float(battery["capacity_kwh"])
    initial_energy = float(battery["initial_energy_kwh"])
    base_max_charge = float(battery["max_charge_kwh_per_hour"])
    base_max_discharge = float(battery["max_discharge_kwh_per_hour"])

    eff_solar = list(solar)
    emin = [base_min] * H
    max_charge = [base_max_charge] * H
    max_discharge = [base_max_discharge] * H
    grid_cap: list[float | None] = [None] * H

    for d in directives:
        t = d.get("directive_type")
        adj = d.get("structured_adjustment") or {}
        adj_hours = adj.get("hours", [])
        if t == "solar_reduction":
            factor = float(adj["factor"])
            for h in adj_hours:
                eff_solar[h] = solar[h] * factor
        elif t == "minimum_battery_reserve":
            level = float(adj["minimum_energy_kwh"])
            for h in adj_hours:
                emin[h] = max(emin[h], level)
        elif t == "no_charge_window":
            for h in adj_hours:
                max_charge[h] = 0.0
        elif t == "no_discharge_window":
            for h in adj_hours:
                max_discharge[h] = 0.0
        elif t == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in adj_hours:
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)
        # no_op / unknown: ignored (guardrails should never let unknown through)

    return ScheduleModel(
        demand=demand,
        tariff=tariff,
        eff_solar=eff_solar,
        emin=emin,
        max_charge=max_charge,
        max_discharge=max_discharge,
        grid_cap=grid_cap,
        capacity=capacity,
        initial_energy=initial_energy,
    )


def _solve_lp(model: ScheduleModel) -> tuple[bool, np.ndarray | None]:
    """Solve the LP for one ScheduleModel. Returns (success, x) where x is the
    flat [grid|solar_used|chg|dis] variable vector (length 4*H) if successful.
    """
    n = 4 * H
    c = np.zeros(n)
    for h in range(H):
        c[h] = model.tariff[h]

    # Equality: energy balance per hour, plus end-of-day neutrality.
    A_eq = np.zeros((H + 1, n))
    b_eq = np.zeros(H + 1)
    for h in range(H):
        A_eq[h, h] = 1.0            # grid
        A_eq[h, H + h] = 1.0        # solar_used
        A_eq[h, 3 * H + h] = 1.0    # dis
        A_eq[h, 2 * H + h] = -1.0   # -chg
        b_eq[h] = model.demand[h]
    for k in range(H):
        A_eq[H, 2 * H + k] = 1.0
        A_eq[H, 3 * H + k] = -1.0
    b_eq[H] = 0.0  # E[23] - E0 == 0  <=>  sum(chg-dis) == 0

    # Inequality: capacity and reserve bounds on the running battery state.
    A_ub = np.zeros((2 * H, n))
    b_ub = np.zeros(2 * H)
    for h in range(H):
        for k in range(h + 1):
            A_ub[h, 2 * H + k] = 1.0
            A_ub[h, 3 * H + k] = -1.0
            A_ub[H + h, 2 * H + k] = -1.0
            A_ub[H + h, 3 * H + k] = 1.0
        b_ub[h] = model.capacity - model.initial_energy
        b_ub[H + h] = model.initial_energy - model.emin[h]

    bounds = []
    for h in range(H):
        gcap = model.grid_cap[h]
        bounds.append((0, gcap if gcap is not None else None))
    for h in range(H):
        bounds.append((0, max(model.eff_solar[h], 0.0)))
    for h in range(H):
        bounds.append((0, max(model.max_charge[h], 0.0)))
    for h in range(H):
        bounds.append((0, max(model.max_discharge[h], 0.0)))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if result.success:
        return True, result.x
    return False, None


def relax_and_solve(
    hours: list[dict], battery: dict, directives: list[dict]
) -> tuple[list[dict], list[str]]:
    """Solve with all directives; if infeasible (should not happen for valid
    organizer scenarios, but hidden-case or LLM-extraction edge cases might
    produce a contradictory combination), progressively relax non-physical
    directive constraints. Physics (balance, capacity, rate limits,
    end-of-day neutrality) is NEVER relaxed - only the directive-imposed caps.

    Returns (hourly_plan, dropped_directive_types).
    """
    directive_types = {d.get("directive_type") for d in directives}

    ladder: list[list[str]] = [
        [],  # level 0: everything applied
        ["max_grid_window"],
        ["max_grid_window", "minimum_battery_reserve"],
        ["max_grid_window", "minimum_battery_reserve", "no_charge_window", "no_discharge_window"],
    ]

    for drop in ladder:
        remaining = [d for d in directives if d.get("directive_type") not in drop]
        model = build_model(hours, battery, remaining)
        ok, x = _solve_lp(model)
        if ok:
            plan = _extract_plan(x, model)
            dropped = sorted(t for t in drop if t in directive_types)
            return plan, dropped

    # Last resort: pure grid-only plan, battery idle throughout. This is
    # always feasible for a request that has already passed schema
    # validation (initial_energy is within [minimum_energy_kwh, capacity]).
    plan = _grid_only_plan(hours, battery)
    return plan, sorted(t for t in directive_types if t != "no_op")


def _extract_plan(x: np.ndarray, model: ScheduleModel) -> list[dict]:
    grid = x[0:H]
    solar_used = x[H : 2 * H]
    chg = x[2 * H : 3 * H]
    dis = x[3 * H : 4 * H]

    plan = []
    energy = model.initial_energy
    for h in range(H):
        c = round(float(chg[h]), ROUND_DP)
        d = round(float(dis[h]), ROUND_DP)
        su = round(float(solar_used[h]), ROUND_DP)
        if c < EPS:
            c = 0.0
        if d < EPS:
            d = 0.0
        if su < EPS:
            su = 0.0

        # Net simultaneous charge+discharge (never required by the cost-only
        # objective, but guard against LP degeneracy) into a single action.
        net = c - d
        if net > EPS:
            action, magnitude = "charge", net
        elif net < -EPS:
            action, magnitude = "discharge", -net
        else:
            action, magnitude = "idle", 0.0

        energy = energy + (magnitude if action == "charge" else 0.0) - (
            magnitude if action == "discharge" else 0.0
        )
        energy = round(energy, ROUND_DP)
        if abs(energy) < EPS:
            energy = 0.0

        # Recompute grid from the exact balance equation using the rounded
        # values, so grid_kwh is never contaminated by rounding elsewhere.
        recomputed_grid = model.demand[h] + magnitude * (1 if action == "charge" else 0) - (
            magnitude if action == "discharge" else 0.0
        ) - su
        recomputed_grid = round(recomputed_grid, ROUND_DP)
        if abs(recomputed_grid) < EPS:
            recomputed_grid = 0.0
        recomputed_grid = max(recomputed_grid, 0.0)

        plan.append(
            {
                "hour": h,
                "grid_kwh": recomputed_grid,
                "solar_used_kwh": su,
                "battery_action": action,
                "battery_kwh": round(magnitude, ROUND_DP),
                "battery_energy_after_kwh": energy,
            }
        )
    return plan


def _grid_only_plan(hours: list[dict], battery: dict) -> list[dict]:
    ordered = sorted(hours, key=lambda h: h["hour"])
    energy = round(float(battery["initial_energy_kwh"]), ROUND_DP)
    plan = []
    for h in ordered:
        demand = round(float(h["demand_kwh"]), ROUND_DP)
        plan.append(
            {
                "hour": h["hour"],
                "grid_kwh": demand,
                "solar_used_kwh": 0.0,
                "battery_action": "idle",
                "battery_kwh": 0.0,
                "battery_energy_after_kwh": energy,
            }
        )
    return plan


# Public alias - used as the absolute last-resort fallback by main.py if
# even the self-replay validator rejects the relaxation-ladder's output
# (should be unreachable, but the SAFE FAILURE requirement means every path
# must terminate in a valid plan rather than a crash).
grid_only_plan = _grid_only_plan


def compute_totals(plan: list[dict], hours: list[dict]) -> tuple[float, float, float]:
    tariff_by_hour = {h["hour"]: float(h["tariff_bdt_per_kwh"]) for h in hours}
    total_grid = sum(p["grid_kwh"] for p in plan)
    total_cost = sum(p["grid_kwh"] * tariff_by_hour[p["hour"]] for p in plan)
    peak_grid = max(p["grid_kwh"] for p in plan)
    return round(total_grid, ROUND_DP), round(total_cost, ROUND_DP), round(peak_grid, ROUND_DP)
