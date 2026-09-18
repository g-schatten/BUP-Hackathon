# GridWise LLM — Smart Campus Energy Optimization

BUP CSE Fest 2026 · Hackathon · Online Preliminary — LLM-Assisted Operator Directive Interpretation.

One HTTP API service that reads a 24-hour campus energy scenario plus 1–3 operator notes,
interprets the notes with an LLM into structured directives, validates them deterministically,
and returns a cost-minimal 24-hour grid/solar/battery schedule that obeys every applicable
directive. Full behavioral spec: `problem-statement/BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf`.

## Architecture

```
POST /optimize-energy
  → request validation (pydantic; malformed/structurally-invalid → 400)
  → LLM interpretation  (app/llm.py)      — the ONLY place a language model is used
  → deterministic guardrails (app/guardrails.py) — repair/clamp/reject untrusted LLM output
  → LP optimizer (app/optimizer.py)       — HiGHS via scipy.optimize.linprog
  → self-replay validator (app/validator.py) — same checks the judge harness runs
  → response
```

- **LLM role**: `app/llm.py` sends the battery spec + all operator notes to **Groq-hosted Llama
  3.3 70B Versatile** (`llama-3.3-70b-versatile`, free tier) in a single forced tool-call
  (`strict: true` JSON schema), converting each note into one of the six supported directive
  types or `no_op`. Groq's LPU inference is used specifically for its very low latency, which
  matters for the p95 <= 5s Performance & Reliability requirement. This is the only place a
  generative model touches the pipeline — `plan_summary` is generated deterministically
  (Section 02 of the Problem Statement explicitly does not require LLM text for that field).
- **Guardrails**: `app/guardrails.py` never trusts the LLM's output directly. It clamps
  out-of-range values (e.g. a reserve above battery capacity), repairs malformed/missing hour
  lists, and falls back to `no_op` only when a directive cannot be made sense of at all. It never
  invents a directive type outside the six supported ones.
- **Optimizer**: `app/optimizer.py` builds a linear program per request (grid / solar_used /
  charge / discharge per hour, energy balance, battery state chain, end-of-day neutrality) and
  solves it with `scipy.optimize.linprog(method="highs")`. If a combination of directives is
  infeasible (should not happen for valid organizer scenarios, but guards against LLM
  extraction edge cases), it progressively relaxes directive-only constraints — physics
  (balance, capacity, rate limits, neutrality) is never relaxed. Verified to reproduce the
  organizer's reference optimal cost exactly (0.00 diff) on all 10 public sample cases.
- **Validator**: `app/validator.py` replays the final plan against every applicable directive
  and every GridWise energy rule before the service responds — this is the "never return an
  invalid plan" safety net (Problem Statement Section 08, SAFE FAILURE).

## Requirements

- Python 3.12
- A free Groq API key with access to `llama-3.3-70b-versatile` (console.groq.com → API Keys).
  Groq's free tier is used deliberately for this project (no billing required); see
  console.groq.com/docs/rate-limits for current free-tier request/token limits.

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key. Never committed; passed at run time only. |
| `GRIDWISE_LLM_MODEL` | No | Overrides the model ID (default `llama-3.3-70b-versatile`). |
| `GRIDWISE_LLM_TIMEOUT_S` | No | Per-attempt LLM request timeout in seconds (default `12.0`). |

No other configuration is required. If `GROQ_API_KEY` is unset or the provider call fails
for any reason, the service does **not** crash — every affected note is safely treated as
`no_op` and a valid (if less directive-aware) schedule is still returned (SAFE FAILURE path).

## Local quickstart (clean environment)

```bash
git clone <this-repo-url>
cd BUP-Hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY=gsk_...   # required for LLM interpretation
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

Sample request (from the public sample pack):

```bash
python3 -c "
import json
d = json.load(open('problem-statement/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'))
print(json.dumps(d['cases'][0]['input']))
" > /tmp/sample1.json

curl -s -X POST http://localhost:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d @/tmp/sample1.json | python3 -m json.tool
```

Run the full public sample pack through the optimizer + validator directly (no HTTP, no LLM —
tests the optimizer/validator logic in isolation using each case's own expected directives):

```bash
python3 tests/run_samples.py
```

Run the public sample pack against a live HTTP endpoint (exercises the full LLM →
guardrails → optimizer pipeline; requires `GROQ_API_KEY` to be set on the server):

```bash
python3 tests/test_endpoint.py http://localhost:8000
```

## Docker

```bash
docker build -t gridwise:latest .
docker run -d --name gridwise --restart unless-stopped -p 8000:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  gridwise:latest
curl -s http://localhost:8000/health
```

### Docker fallback image (pullable)

```bash
docker pull ghcr.io/<owner>/gridwise:<tag>
docker run -d --name gridwise --restart unless-stopped -p 80:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  ghcr.io/<owner>/gridwise:<tag>
```

No secrets are baked into the image; `GROQ_API_KEY` is supplied only at `docker run` time.

## Deployed endpoint

- Base URL: `http://gridwise-fop62.centralindia.cloudapp.azure.com`
- `GET /health`
- `POST /optimize-energy`

## Dependencies / libraries credited

- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — HTTP service
- [scipy](https://scipy.org/) (`linprog`, HiGHS solver) — the optimization engine
- [pydantic](https://docs.pydantic.dev/) — request/response schema validation
- [Groq Python SDK](https://github.com/groq/groq-python) — LLM interpretation (Llama 3.3 70B Versatile, free tier)

## Known limitations

- The LP relaxation ladder in `app/optimizer.py::relax_and_solve` only engages if a directive
  combination is infeasible; the Problem Statement guarantees organizer scoring scenarios are
  feasible, so this path is a defensive safety net rather than expected behavior.
- `plan_summary` is a deterministic template, not LLM-generated text (by design — see
  Architecture above; the Problem Statement explicitly excludes this field from the LLM
  requirement).
- The in-process LLM response cache (keyed by exact note text + battery capacity) only helps
  repeated identical notes within one running process; it does not persist across restarts.

## Secret handling

No API keys, tokens, or `.env` files are committed to this repository. `GROQ_API_KEY` is
read from the environment only. Application logs never print request/response bodies containing
secrets, and unhandled server errors return a generic `500` with no stack trace to the client.
