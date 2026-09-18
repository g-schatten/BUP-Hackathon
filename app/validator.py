"""Deterministic replay validator.

Mirrors exactly what the hidden judge harness is documented to check
(Problem Statement Sections 09 & 11, Participant Guide Section 09):
  - hourly_plan has exactly 24 unique hours 0..23
  - all required numeric values are finite and non-negative
  - solar_used_kwh never exceeds effective solar for that hour
  - energy-balance equation holds every hour
  - battery transitions, capacity, minimum energy, and rate limits are valid
  - every applicable directive (solar_reduction / minimum_battery_reserve /
    no_charge_window / no_discharge_window / max_grid_window) is respected
  - final battery energy equals initial battery energy
  - reported totals match values recalculated from hourly_plan

This is used twice: (1) as a final self-check before the service responds
(SAFE FAILURE requirement - never return an invalid plan), and (2) by the
offline test harness to score our own output against the public sample pack.
"""
from __future__ import annotations

import math

from app.optimizer import H, build_model

TOL = 0.01  # Section 11.5 numeric tolerance


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def validate_plan(
    hours: list[dict],
    battery: dict,
    directives: list[dict],
    plan: list[dict],
    reported_totals: tuple[float, float, float] | None = None,
) -> list[str]:
    """Return a list of human-readable violation strings; empty list == valid."""
    errors: list[str] = []

    # --- structural: exactly 24 unique hours 0..23 -------------------------
    plan_hours = [p["hour"] for p in plan]
    if sorted(plan_hours) != list(range(H)):
        errors.append("hourly_plan must contain exactly 24 unique hours 0..23")
        return errors  # further checks are meaningless without this

    plan_by_hour = {p["hour"]: p for p in plan}
    model = build_model(hours, battery, directives)
    demand_by_hour = {h["hour"]: float(h["demand_kwh"]) for h in hours}

    energy = model.initial_energy
    for h in range(H):
        p = plan_by_hour[h]
        grid = p["grid_kwh"]
        solar_used = p["solar_used_kwh"]
        action = p["battery_action"]
        mag = p["battery_kwh"]
        after = p["battery_energy_after_kwh"]

        for name, val in (
            ("grid_kwh", grid),
            ("solar_used_kwh", solar_used),
            ("battery_kwh", mag),
            ("battery_energy_after_kwh", after),
        ):
            if val is None or not math.isfinite(val):
                errors.append(f"hour {h}: {name} is not finite")
                continue
            if name != "battery_energy_after_kwh" and val < -TOL:
                errors.append(f"hour {h}: {name} is negative ({val})")

        if action not in ("charge", "discharge", "idle"):
            errors.append(f"hour {h}: invalid battery_action '{action}'")
            continue

        if action == "idle" and abs(mag) > TOL:
            errors.append(f"hour {h}: battery_action idle but battery_kwh={mag} != 0")

        chg = mag if action == "charge" else 0.0
        dis = mag if action == "discharge" else 0.0

        if solar_used > model.eff_solar[h] + TOL:
            errors.append(
                f"hour {h}: solar_used_kwh {solar_used} exceeds effective solar {model.eff_solar[h]}"
            )

        if chg > model.max_charge[h] + TOL:
            errors.append(f"hour {h}: charge {chg} exceeds max_charge_kwh_per_hour {model.max_charge[h]}")
        if dis > model.max_discharge[h] + TOL:
            errors.append(f"hour {h}: discharge {dis} exceeds max_discharge_kwh_per_hour {model.max_discharge[h]}")

        if model.grid_cap[h] is not None and grid > model.grid_cap[h] + TOL:
            errors.append(f"hour {h}: grid_kwh {grid} exceeds max_grid_window cap {model.grid_cap[h]}")

        demand = demand_by_hour[h]
        balance_lhs = grid + solar_used + dis
        balance_rhs = demand + chg
        if not _close(balance_lhs, balance_rhs):
            errors.append(
                f"hour {h}: energy balance violated ({balance_lhs} != {balance_rhs}: "
                f"grid={grid} solar_used={solar_used} discharge={dis} demand={demand} charge={chg})"
            )

        expected_after = energy + chg - dis
        if not _close(after, expected_after):
            errors.append(
                f"hour {h}: battery_energy_after_kwh {after} does not follow from the prior "
                f"state ({energy}) and this hour's action (expected {expected_after})"
            )
        if after < model.emin[h] - TOL:
            errors.append(f"hour {h}: battery_energy_after_kwh {after} below required minimum {model.emin[h]}")
        if after > model.capacity + TOL:
            errors.append(f"hour {h}: battery_energy_after_kwh {after} exceeds capacity {model.capacity}")

        energy = after

    if not _close(energy, model.initial_energy):
        errors.append(
            f"end-of-day battery neutrality violated: final energy {energy} != initial {model.initial_energy}"
        )

    if reported_totals is not None:
        r_grid, r_cost, r_peak = reported_totals
        calc_grid = sum(plan_by_hour[h]["grid_kwh"] for h in range(H))
        tariff_by_hour = {h["hour"]: float(h["tariff_bdt_per_kwh"]) for h in hours}
        calc_cost = sum(plan_by_hour[h]["grid_kwh"] * tariff_by_hour[h] for h in range(H))
        calc_peak = max(plan_by_hour[h]["grid_kwh"] for h in range(H))
        if not _close(r_grid, calc_grid):
            errors.append(f"total_grid_kwh {r_grid} does not match recalculated value {calc_grid}")
        if not _close(r_cost, calc_cost):
            errors.append(f"total_cost_bdt {r_cost} does not match recalculated value {calc_cost}")
        if not _close(r_peak, calc_peak):
            errors.append(f"peak_grid_kwh {r_peak} does not match recalculated value {calc_peak}")

    return errors
