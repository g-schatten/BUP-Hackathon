"""GridWise LLM-assisted energy optimization API.

Pipeline (Problem Statement Section 03):
  request validation -> LLM note interpretation -> deterministic guardrails
  -> LP optimization -> self-replay validation -> response

Endpoints (Section 06): GET /health, POST /optimize-energy.
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.guardrails import apply_guardrails
from app.llm import interpret_notes
from app.optimizer import compute_totals, grid_only_plan, relax_and_solve
from app.schemas import HealthResponse, OptimizeEnergyRequest
from app.validator import validate_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM Optimizer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return HealthResponse().model_dump()


def _build_summary(
    directive_interpretation: list[dict],
    relaxed_types: list[str],
    total_cost: float,
    total_grid: float,
    peak_grid: float,
) -> str:
    """Deterministic (non-LLM) human-readable summary. The Problem Statement
    explicitly does not require plan_summary to come from the LLM, and using
    an LLM ONLY for this field would not satisfy the LLM requirement anyway -
    so this stays cheap and fast.
    """
    applied = [d for d in directive_interpretation if d["applies"]]
    ignored = len(directive_interpretation) - len(applied)
    parts = []
    if applied:
        types = ", ".join(sorted({d["directive_type"] for d in applied}))
        parts.append(f"Applied {len(applied)} operator directive(s) ({types})")
    else:
        parts.append("No operator directives applied")
    if ignored:
        parts.append(f"ignored {ignored} unrelated note(s) as no_op")
    if relaxed_types:
        parts.append(
            f"relaxed conflicting directive(s) ({', '.join(relaxed_types)}) to keep the schedule physically valid"
        )
    parts.append(
        f"total grid cost {total_cost:.2f} BDT over 24h, {total_grid:.2f} kWh imported, peak {peak_grid:.2f} kWh/h"
    )
    return "; ".join(parts) + "."


@app.post("/optimize-energy")
async def optimize_energy(request: Request) -> JSONResponse:
    raw_body = await request.body()
    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "malformed JSON body"})

    try:
        req = OptimizeEnergyRequest.model_validate(data)
    except ValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "request does not match the required schema", "detail": exc.errors()[:10]},
        )
    except Exception:
        return JSONResponse(status_code=400, content={"error": "structurally invalid request"})

    scenario_id = req.scenario_id
    t0 = time.monotonic()
    try:
        hours = [h.model_dump() for h in req.hours]
        battery = req.battery.model_dump()

        raw_directives = interpret_notes(req.operator_notes, battery["capacity_kwh"])
        directive_interpretation = apply_guardrails(
            req.operator_notes, raw_directives, battery["capacity_kwh"]
        )

        applying = [d for d in directive_interpretation if d["applies"]]
        plan, dropped_types = relax_and_solve(hours, battery, applying)
        total_grid, total_cost, peak_grid = compute_totals(plan, hours)

        actually_applied = [d for d in applying if d["directive_type"] not in dropped_types]
        errors = validate_plan(
            hours, battery, actually_applied, plan,
            reported_totals=(total_grid, total_cost, peak_grid),
        )

        if errors:
            # Should be unreachable given relax_and_solve's guarantees, but the
            # SAFE FAILURE requirement means we never return a plan we know is
            # invalid. Fall back to the always-feasible grid-only schedule.
            logger.error("scenario %s: self-replay failed, falling back to grid-only. errors=%s", scenario_id, errors)
            plan = grid_only_plan(hours, battery)
            total_grid, total_cost, peak_grid = compute_totals(plan, hours)
            dropped_types = sorted({d["directive_type"] for d in applying})

        summary = _build_summary(directive_interpretation, dropped_types, total_cost, total_grid, peak_grid)

        payload = {
            "scenario_id": scenario_id,
            "directive_interpretation": directive_interpretation,
            "hourly_plan": plan,
            "total_grid_kwh": total_grid,
            "total_cost_bdt": total_cost,
            "peak_grid_kwh": peak_grid,
            "plan_summary": summary,
        }
        elapsed = time.monotonic() - t0
        logger.info("scenario %s handled in %.2fs (dropped=%s)", scenario_id, elapsed, dropped_types)
        return JSONResponse(status_code=200, content=payload)

    except Exception:
        logger.exception("scenario %s: unhandled internal error", scenario_id)
        return JSONResponse(status_code=500, content={"error": "internal server error"})
