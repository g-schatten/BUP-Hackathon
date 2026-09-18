# 3-Minute Solution Video — Script & Shot List

**Hard cap 3:00** — the guide treats over-length as a fail on this criterion, so the margin matters.

Narration is **398 words** (measured, not estimated). Actual runtime depends on your pace:

| Your pace | Narration alone | + ~20s of pauses/transitions |
|---|---|---|
| 135 wpm (slow) | 2:56 | **3:16 — too long** |
| 150 wpm (normal) | 2:39 | 2:59 — cutting it fine |
| 165 wpm (brisk) | 2:24 | **2:44 — safe** |

**So: read at a brisk-but-natural pace and keep screen transitions tight.** Time your first take.
If it lands over 2:50, use the cut list in Delivery Notes at the bottom rather than speeding up.

**What the rubric requires this video to cover** (Participant Guide §02, §08, §10 — scored only as
a tie-break, but reviewers compare exactly these four things):

| # | Requirement | Covered in |
|---|---|---|
| 1 | Problem understanding | Section 1 |
| 2 | Architecture overview | Section 2 |
| 3 | LLM → deterministic guardrails → optimizer flow | Sections 2 & 3 |
| 4 | How the solution is run and tested | Section 4 |

---

## Before you hit record

- [ ] Terminal at `~/Desktop/BUP-Hackathon`, font size bumped up (readable at 1080p)
- [ ] Editor open with `app/llm.py`, `app/guardrails.py`, `app/optimizer.py`, `app/validator.py` in tabs
- [ ] Browser tab on `http://gridwise-fop62.centralindia.cloudapp.azure.com/health` — **http**, not https
- [ ] Regenerate the demo payload (`/tmp` clears on reboot — run this first, every time):
      ```bash
      cd ~/Desktop/BUP-Hackathon && python3 -c "
      import json
      d = json.load(open('problem-statement/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'))
      print(json.dumps(d['cases'][0]['input']))" > /tmp/sample1.json
      ```
- [ ] Warm the LLM cache so the live demo is fast and reliable:
      `.venv/bin/python tests/test_endpoint.py http://gridwise-fop62.centralindia.cloudapp.azure.com`
- [ ] Section 4's commands copied somewhere you can paste from mid-recording

---

## Section 1 — The problem (0:00–0:35) · *Requirement 1*

**On screen:** the Problem Statement PDF, then the sample JSON showing `operator_notes`.

> "BUP's campus runs on grid power, rooftop solar and a battery. We get 24 hours of demand, solar
> and tariff — plus one to three operator notes in plain English.
>
> Those notes aren't math. 'An eighty percent reduction in solar from 1 to 3 PM' must become hours
> 13 and 14, factor 0.2 — the fraction *remaining*, not removed. And distractors like 'the cafeteria
> menu changes tomorrow' must be ignored, not invented into a rule.
>
> So: understand the notes, apply them, and return the cheapest valid 24-hour schedule."

---

## Section 2 — Architecture (0:35–1:00) · *Requirements 2 & 3*

**On screen:** the architecture diagram in `README.md`, highlighting each stage as you name it.

> "Four stages — and the key principle is that the LLM's output is treated as untrusted until
> deterministic code has checked it.
>
> Schema validation rejects malformed requests. The LLM interprets every note. Deterministic
> guardrails validate and repair that output. A linear-programming optimizer builds the schedule,
> and a self-replay validator re-checks the finished plan before we respond."

---

## Section 3 — Implementation (1:00–2:00) · *Requirement 3, in depth*

**On screen:** cut between the four files as you mention each, ~13s per file.

**`app/llm.py`:**
> "The LLM is Groq-hosted `gpt-oss-20b` — all notes in one forced tool call, strict JSON schema,
> temperature zero. The battery spec is in the prompt too, because 'keep fifty percent of capacity
> in reserve' only becomes kilowatt-hours if the model knows the capacity."

**`app/guardrails.py`:**
> "Guardrails don't trust that output. Hours are de-duplicated, sorted and range-checked; the solar
> factor clamped to zero-to-one; an over-capacity reserve clamped down. Crucially we repair rather
> than discard — a repaired directive still earns credit; dropping it to `no_op` scores nothing.
> And unsupported types are never invented."

**`app/optimizer.py`:**
> "The optimizer is a linear program solved with SciPy's HiGHS — energy balance, battery limits and
> end-of-day neutrality as constraints, minimising cost. Directives simply modify that model."

**`app/validator.py`:**
> "Then we replay our own plan against every judge rule before responding, falling back to a
> guaranteed-valid schedule if anything fails. If the provider dies, notes degrade to `no_op` and a
> valid schedule still comes back."

---

## Section 4 — Run & test (2:00–2:30) · *Requirement 4*

**On screen:** live terminal. Run these for real — don't show stills.

**Shot A:**
```bash
curl -s http://gridwise-fop62.centralindia.cloudapp.azure.com/health
```
> "Deployed on an Azure VM in Docker, publicly reachable — health returns ok."

**Shot B:**
```bash
curl -s -X POST http://gridwise-fop62.centralindia.cloudapp.azure.com/optimize-energy \
  -H 'Content-Type: application/json' -d @/tmp/sample1.json | python3 -m json.tool | head -30
```
> "A live request: the solar-cleaning note became `solar_reduction`, hours 12 and 13, factor 0.25 —
> and the distractor came back `no_op`."

**Shot C:**
```bash
.venv/bin/python tests/run_samples.py
.venv/bin/python tests/test_endpoint.py http://gridwise-fop62.centralindia.cloudapp.azure.com
```
> "Two harnesses: one tests the optimizer alone against all ten public cases; the other drives the
> full live pipeline over HTTP, scoring interpretation, validity, cost and latency."

---

## Section 5 — Results & close (2:30–2:45)

**On screen:** the final summary lines of the test output.

> "Ten out of ten on interpretation, every plan valid, and our cost matches the organisers'
> reference optimum exactly — quality ratio 1.0, at about a second median latency. Setup and the
> Docker fallback are in the README. Thanks for watching."

---

## Delivery notes

- **The one sentence that matters most** to a tie-break reviewer is in Section 2: *"the LLM's output
  is treated as untrusted until deterministic code has checked it."* Say it clearly — that
  separation is exactly what they're grading.
- Narrate the *intent* while code is on screen; don't read code aloud.
- One take is fine. Production quality is explicitly not judged — clarity is.
- **If you overrun**, cut in this order: (1) the setup sentence in Section 5, (2) the last sentence
  of the `validator.py` paragraph, (3) the `optimizer.py` paragraph down to "It's a linear program
  solved with SciPy's HiGHS; directives modify the model."
