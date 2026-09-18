# GridWise LLM — Execution Plan (BUP CSE Fest 2026 Preliminary)

## 0. What is being scored (100 pts, automated)

| # | Category | Pts | Where it's won |
|---|---|---|---|
| 1 | LLM Directive Interpretation | 25 | 5 relevance/no_op + 5 type + 5 hours + 5 numeric/shape + 5 paraphrase robustness |
| 2 | Directive Application & Constraint Correctness | 25 | 10 apply ground-truth directives + 5 balance/solar + 5 battery bounds/rates + 5 action consistency/neutrality |
| 3 | Optimization Quality | 10 | `10 x avg(min(1, optimal/ours))`, **only on cases already judged valid** |
| 4 | API Contract & Schema | 10 | endpoints, 400/422/500, interpretation order/types, response schema, scenario_id echo |
| 5 | Performance & Reliability | 10 | 2 health + 3 p95 latency (**≤5s for full marks**) + 3 stability + 2 safe failure/secrets |
| 6 | Deployment & Docker Fallback | 10 | 3 live URL + 4 pullable image reaching /health + 2 clean startup + 1 no judge debugging |
| 7 | Documentation & Local Reproducibility | 10 | README quickstart, env vars, sample test, architecture, docker, deps/limits |

**50 of 100 points ride on notes → directives → schedule.** Cost optimization is only 10.
Ground truth beats cost: judge replays our plan against **its own** directives, not ours.

## 1. Key findings from the spec (already verified)

- **The optimizer is a pure LP and I have solved it.** Variables per hour `h`: `grid[h]≥0`, `solar_used[h]∈[0, eff_solar[h]]`, `chg[h]∈[0,maxC]`, `dis[h]∈[0,maxD]`.
  - balance: `grid + solar_used + dis = demand + chg`
  - `E[h] = E0 + Σ_{k≤h}(chg−dis)`, with `emin[h] ≤ E[h] ≤ capacity`, and `E[23] = E0`
  - objective `min Σ grid[h]·tariff[h]`
  - **Validated: scipy/HiGHS reproduces the organizer reference cost exactly on 10/10 public samples (diff 0.00).** Optimization Quality should be a clean 10/10 whenever interpretation is right.
- Simultaneous charge+discharge is never needed (no round-trip losses) → net the LP solution into exactly one of `charge`/`discharge`/`idle`.
- Directive → math mapping: `solar_reduction` scales `eff_solar`; `minimum_battery_reserve` raises `emin[h]` to `max(base, directive)`; `no_charge_window` sets `maxC=0`; `no_discharge_window` sets `maxD=0`; `max_grid_window` caps `grid[h]`.
- **Interpretation traps observed in the public pack:**
  - Windows are start-inclusive / end-exclusive: "6 PM until 9 PM" → `[18,19,20]`; "1 PM to 3 PM" → `[13,14]`; "between 11 AM and 2 PM" → `[11,12,13]`.
  - `factor` = fraction **remaining**: "80% reduction" → 0.2; "drop to 20%" → 0.2; "one-fifth" → 0.2; "about half" → 0.5; "roughly 25%" → 0.25.
  - **Relative reserves must be resolved against battery params**: SAMPLE-03 "at least 50% of battery capacity" with `capacity=200` → `minimum_energy_kwh: 100`. → the battery object MUST be in the LLM prompt.
  - Distractors are campus-admin chatter (cafeteria menu, library hours, seminar booking, registration deadline) → `no_op`, `applies=false`, `structured_adjustment=null`.
  - Hidden extras to prompt for: "all day/throughout the day" → `[0..23]`; wrap-around windows ("10 PM until 2 AM") → sorted ascending `[0,1,22,23]`; 24h clock ("13:00–15:00"); word numbers ("one until three").

## 2. Architecture

```
POST /optimize-energy
  → request validation (pydantic, strict)          -> 400 on malformed
  → note interpretation (LLM, forced JSON schema)  ─┐ per-note, one batched call
  → deterministic guardrails + repair              ─┘ clamp/sort/dedupe, never invent a type
  → build LP constraints from directives
  → HiGHS LP solve  (fallback: slack-penalty LP if infeasible)
  → schedule normalization (net battery, exact rounding, recompute totals)
  → SELF-REPLAY VALIDATOR (same checks the judge runs)  -> if fail, safe fallback plan
  → response
```

**Stack:** Python 3.12 · FastAPI + uvicorn · scipy HiGHS (`linprog`) · `groq` SDK (**Llama 3.3 70B Versatile**, model ID `llama-3.3-70b-versatile`, free tier) · Docker · Azure VM.

**Decision change (2026-09-18): switched from Anthropic to Groq at the user's request for a
free option.** Groq's free tier hosts open models (Llama 3.3 70B Versatile, gpt-oss-20b/120b,
etc.) on custom LPU inference hardware at very low latency (~280 tok/s for the 70B model per
console.groq.com/docs/models, confirmed live) with **no billing required**. This is a genuine
upgrade on the latency dimension versus the original Anthropic plan, at zero cost, with the
trade-off that free-tier rate limits apply (check console.groq.com/docs/rate-limits before the
event — if the hidden test volume is large, confirm the limit comfortably covers it, or fall
back to a paid tier / secondary key). The Groq chat.completions API is OpenAI-compatible
(`tools`/`tool_choice`/`response_format`), verified directly against the installed `groq` 1.7.0
SDK's type definitions rather than assumed from memory.

### LLM layer rules — MEASURED against the live account, not assumed

**Model catalog drifted from what the public docs page said.** `llama-3.3-70b-versatile` (the
originally planned model) returned `404 model_not_found` on this account — Groq's currently
available chat models here are `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and
`qwen/qwen3.8-27b` (confirmed via an authenticated `GET /openai/v1/models` call, which is ground
truth over any cached docs page). **In use: `openai/gpt-oss-20b`.**

**The real free-tier constraint is TPM (tokens/minute), not RPM.** Every real chat model on this
account is capped at **8000 tokens/minute** (confirmed via the `x-ratelimit-limit-tokens`
response header on all three candidates — it's an account-level ceiling, not model-specific).
`groq/compound`/`compound-mini` have a much higher TPM (70000) but explicitly do not support
custom tool definitions ("Custom tools are not supported at this time" per Groq's own compound
docs), so they can't run our extraction tool at all. Mitigations applied, in order of impact:
1. **Shrank the system prompt ~60%** (1320 → ~540 estimated tokens): fewer, denser few-shot
   examples, no restated rules. This alone roughly doubled how many calls fit in one TPM window
   before throttling.
2. **`reasoning_effort="low"`** on the gpt-oss/qwen3 reasoning models — the Groq-documented
   default ("medium") produced wildly variable reasoning-token counts (12–20s wall-clock on some
   public cases even before any rate-limit was involved).
3. **`temperature=0`** — not a token-budget fix, but cut real accuracy flakiness observed in
   testing (a distractor note flipped between `no_op` and a real directive across otherwise
   identical runs); determinism also helps the paraphrase-robustness sub-score.
4. **`max_completion_tokens=1024`** (down from 2048) as a hard ceiling on worst-case output spend.
5. In-process cache by `sha256(notes + battery_capacity)` — confirmed empirically: a warm-cache
   rerun of all 10 public cases came back at p95 0.38s with zero new API calls.

**Result on the 10 public samples, cold cache, `openai/gpt-oss-20b`:** 10/10 interpretation
correct, 10/10 valid plans, cost-quality ratio 1.0000, **p50 latency ~1.1–1.5s, p95 ~8–10s**
(the p95 tail is exactly the TPM throttling kicking in after ~6-7 back-to-back novel calls, each
throttled wait plus retry still lands safely under the judge's 30s hard timeout in every observed
case, but this is the free tier's one real remaining risk under bursty hidden-test traffic — see
"Known limitations" in README.md). `tool_choice` forced to the one function, **`strict: true`**
on its schema, `json.loads` + pydantic re-validation of the arguments (Groq's own docs note the
model "does not always generate valid JSON"), `max_retries=1` on transport/429 errors.
- Prompt carries: the six directive types with exact `structured_adjustment` shapes, the hour-window convention, factor semantics, the battery object (for % reserves), and one compact example per type + a distractor.
- **Guardrails (deterministic, after the model):** type ∈ enum; exactly one entry per note index, ascending; hours unique ints 0–23 sorted; `0 ≤ factor ≤ 1`; `0 ≤ reserve ≤ capacity`; `max_grid_kwh ≥ 0` finite; `applies=true` iff not `no_op`; `structured_adjustment=null` iff `no_op`. **Repair, don't discard** — a repaired directive still earns application credit; a dropped one earns nothing.
- **Fallback interpreter (regex/heuristic) runs only if the provider errors or output is unusable.** Documented as a degraded path, never the primary — sole phrase-matching is explicitly non-compliant.
- Cache by `sha256(note)` → interpretation, in-process. Hidden sets reuse phrasings; this protects p95 latency.

### Never return an invalid plan
1. Hard-constrained LP.
2. If infeasible → re-solve with slack variables on *directive* constraints only (huge penalty), so physics/battery rules always hold.
3. If the self-replay validator still fails → grid-only plan (`grid=demand`, battery idle, solar_used=0) which is always valid unless a reserve above `E0` is active.
4. Never 500 on a well-formed request; never leak stack traces or keys.

### Rounding discipline (cheap points, easy to lose)
Round `solar_used`, `chg`, `dis` to 3dp → recompute `E[h]` by chain from rounded values → derive `grid[h] = demand + chg − dis − solar_used` exactly → clamp |x|<1e-9 to 0 → recompute `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` **from the final plan array**. Totals must match the judge's recalculation within 0.01.

## 3. Timeline (4-hour window, 7:00–11:00 PM)

| Time | Milestone | Why in this order |
|---|---|---|
| 0:00–0:25 | New private repo, FastAPI skeleton, `/health`, pydantic schemas, **Azure VM up and serving a stub `/health` on a public IP** | Deployment is the #1 risk; get the URL alive before writing logic |
| 0:20–0:50 | LP optimizer + replay validator, offline harness over the 10 public cases using *expected* directives | Proves 10/10 optimization independent of the LLM |
| 0:50–1:35 | LLM interpreter + prompt + guardrails; score interpretation against all 10 public expected outputs | The 25-point block |
| 1:35–2:00 | Wire end-to-end, run the full harness (interpretation + validity + cost) against the live URL | First green end-to-end |
| 2:00–2:30 | Robustness: malformed JSON → 400, provider outage → fallback, infeasible → slack LP, caching, timeouts, latency measurement | Performance & Reliability 10 |
| 2:30–3:00 | Push image to **GHCR** with an exact tag, `docker pull`+`run` verified clean, redeploy on the Azure VM, external smoke test from phone/other network | Deployment 10 |
| 3:00–3:30 | README (quickstart, env var *names* only, model/provider, LLM role, guardrails, solver, curl examples, sample-test command, deps, limitations) | Documentation 10 |
| 3:30–3:50 | 3-min video: problem → architecture → LLM/guardrail/optimizer flow → run & test demo | Tie-break only, but cheap insurance |
| 3:50–4:00 | Final checklist, make repo public after deadline, submit all 5 artifacts | — |

**If on a team:** one person owns deploy+Docker+README, one owns the LP+validator, one owns the LLM prompt+guardrails. They only meet at 1:35.

## 4. Test harness (build this early, it pays for itself)
`tests/run_samples.py` — for each public case: POST to a target URL, then
1. diff `directive_interpretation` against expected (type/applies/hours/numbers, ignoring explanation text),
2. replay `hourly_plan` under the **expected** directives with every judge rule,
3. compare our cost to the reference cost → `min(1, ref/ours)`,
4. print a per-case scorecard and p95 latency.
Plus a paraphrase suite: hand-written re-wordings of each directive type (24h clock, word numbers, "reduction" vs "drops to", "%-of-capacity" reserves, wrap-around windows, all-day) to test robustness the public pack doesn't cover.

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| VM reboot / container crash mid-judging | `--restart unless-stopped` on the container; VM stays running (no autoshutdown policy) for the whole window |
| Azure credit burn | `Standard_B2s` ≈ $0.04/hr → ~$0.20 for the round; `az vm deallocate` after submission closes |
| LLM latency blows p95 ≤ 5s | Small fast model, single batched call, note-level cache, 8s client timeout + fallback |
| Provider quota/outage during judging | Retry + secondary provider key + deterministic fallback interpreter |
| Our interpretation makes the LP infeasible | Slack-penalty LP; never return a 500 |
| Rounding drift breaks balance/totals | Derive `grid` from the balance equation last; self-replay before responding |
| Repo policy | New repo created **after** question reveal, private during the round, public right after the deadline; no secrets committed, none baked into the image |

## 6. Azure deployment runbook — EXECUTED, live

**Status: done.** Public endpoint is live: `http://gridwise-fop62.centralindia.cloudapp.azure.com`
(`GET /health` -> `{"status":"ok"}`, `POST /optimize-energy` verified externally).

**Deviation from the original plan, and why:** this Azure for Students subscription carries an
org policy (`sys.regionrestriction`) that hard-restricts deployable regions to exactly
`eastasia`, `japanwest`, `indonesiacentral`, `uaenorth`, `centralindia` — `southeastasia` (the
original plan) is rejected outright (`RequestDisallowedByAzure`). Discovered via:
```bash
az policy assignment list --disable-scope-strict-match -o json | jq '.[].parameters.listOfAllowedLocations.value'
```
Picked **`centralindia`** (best latency to Dhaka among the allowed set). Separately,
`Standard_B2s` reported `SkuNotAvailable` (capacity restrictions) specifically in `centralindia`
even though `az vm list-sizes` listed it — used **`Standard_B2as_v2`** instead (same 2 vCPU / 4 GB
shape, AMD-based, Bsv2 family, quota headroom confirmed with `az vm list-usage`). If re-running
this in a fresh subscription, check both constraints first — the allowed-locations policy and
per-size capacity — rather than assuming the originally-planned region/size will work.

Public endpoint = Azure Linux VM with a **DNS name label**, container published on port 80.
Docker fallback image = **GHCR** (free, anonymous public pull — Azure Container Registry Basic does *not* support anonymous pull, so it is the wrong choice for a judge-pullable fallback).

```bash
# 1. one-time: resource group + VM with a stable public DNS name (ACTUAL commands used)
az group create -n gridwise-rg -l centralindia

az vm create -g gridwise-rg -n gridwise-vm \
  --image "Canonical:ubuntu-24_04-lts:server:latest" --size Standard_B2as_v2 \
  --admin-username azureuser --generate-ssh-keys \
  --public-ip-sku Standard \
  --public-ip-address-dns-name gridwise-fop62 \
  --custom-data cloud-init.yaml   # installs Docker via get.docker.com at boot
az vm open-port -g gridwise-rg -n gridwise-vm --port 80 --priority 900

# endpoint => http://gridwise-fop62.centralindia.cloudapp.azure.com
```

Deploy/redeploy in this session's workflow (no GHCR needed for the live endpoint - source is
rsynced directly and built on the VM; GHCR is only for the separate fallback-image deliverable):
```bash
rsync -az --delete app requirements.txt Dockerfile .dockerignore azureuser@$FQDN:~/gridwise-app/
ssh azureuser@$FQDN "cd ~/gridwise-app && sudo docker build -t gridwise:v1 ."
ssh azureuser@$FQDN "sudo docker rm -f gridwise 2>/dev/null; sudo docker run -d --name gridwise --restart unless-stopped -p 80:8000 -e GROQ_API_KEY=\"\$GROQ_API_KEY\" gridwise:v1"
```

```bash
# 2. each redeploy (from the VM)
docker pull ghcr.io/<user>/gridwise:<tag>
docker rm -f gridwise 2>/dev/null
docker run -d --name gridwise --restart unless-stopped -p 80:8000 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  ghcr.io/<user>/gridwise:<tag>
curl -s http://localhost/health
```

Rules to hold to:
- **The key never enters the image or the repo** — only `-e GROQ_API_KEY` at run time; README documents the *name* only.
- Keep a 2 vCPU / 4 GB shape (`Standard_B2as_v2` in use) — a 1 GB `B1s`-class size is tight for scipy + uvicorn and risks an OOM mid-judging. ~$0.03-0.04/hr, well inside the student credit.
- Give the VM a DNS name label at create time; if time allows late in the round, put Caddy in front for automatic HTTPS on that hostname (plain HTTP on port 80 is acceptable and is the default path).
- No auto-shutdown schedule on the VM. `az vm deallocate` only after the evaluation window closes.
- Tag the image with an immutable tag (e.g. `v1`, `v2`) and submit the exact one that is running.

## 7. Open decisions
Live endpoint is up; the two remaining blockers are the `GROQ_API_KEY` (LLM interpretation is currently in safe-failure no_op mode without it) and a GitHub username/PAT to push the GHCR fallback image (not yet built - `gh auth status` shows no logged-in host in this environment).
