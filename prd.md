# PRD: Hybrid Token-Efficient Routing Agent

**Track:** Track 1 — Hybrid Token-Efficient Routing Agent (AI Agent Track, Beginner Friendly)
**Author:** Krishanth
**Status:** Draft for build
**Timeline:** 4 days to submission

---

## 1. Problem Statement

The agent must complete a fixed set of tasks (revealed at kickoff) autonomously, choosing per-task between:
- **Local inference** (zero cost toward score)
- **Remote inference via Fireworks AI** (costs tokens, counts toward score)

The scoring function rewards **minimum total tokens spent**, subject to **not falling below an accuracy threshold**. This is fundamentally a **constrained optimization problem**, not a classification problem: every routing decision trades expected accuracy gain against expected token cost, under a shared, depleting budget.

The core failure mode to avoid: trusting the local model's self-reported confidence as a signal of correctness. LLMs are poorly calibrated — stated confidence does not reliably predict correctness. The system must use **behavioral and statistical signals**, not self-report, to decide when escalation is justified.

---

## 2. Goals

### Primary goals
1. Maximize accuracy-per-token across the full task set (i.e., minimize total tokens spent while staying above the accuracy floor).
2. Make routing decisions **autonomously and in real time**, with no human in the loop.
3. Be robust to a small/unknown task count — the system should not require large amounts of data to calibrate.

### Non-goals (out of scope for this build)
- Fine-tuning the local or remote model (prompt-based routing is explicitly scored equally, and is lower-risk given the timeline).
- Optimizing for latency — only token count and accuracy are judged.
- General-purpose routing beyond the task categories seen in this competition.

---

## 3. Success Metrics

| Metric | Target |
|---|---|
| Total remote (Fireworks) tokens used across all tasks | Minimized, subject to accuracy floor |
| Task accuracy (however scored: exact-match / LLM-judge / rubric — confirm at kickoff) | ≥ threshold (assume threshold unknown until kickoff; build with margin) |
| % of tasks resolved without any remote call | Track as internal proxy metric; higher is directionally good but not directly scored |
| Router decision latency | Not scored, but should not be pathologically slow (keep under a few seconds per task) |

---

## 4. System Overview

```
                ┌─────────────────────┐
   Task in ───▶ │   Local Model Pass    │
                │ (always runs first)  │
                └──────────┬────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Trust Evaluator     │
                │ (multi-signal check) │
                └──────────┬────────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
     Trusted (no escalate)      Not trusted (escalate)
              │                          │
              ▼                          ▼
        Return local answer     Call Fireworks remote model
                                        │
                                        ▼
                                Return remote answer
                                        │
                                        ▼
                          Update per-category history
                          (feeds Budget-Aware Threshold Adjuster)
```

The system has three logical components:

1. **Local Executor** — runs every task through a local model first, always. Local tokens are free, so there is no reason to skip this step.
2. **Trust Evaluator** — decides whether the local answer is good enough to return, using multiple independent, non-self-reported signals (Section 5).
3. **Budget-Aware Threshold Adjuster** — maintains running state (tokens spent, tasks remaining, per-category historical reliability) and dynamically tunes how strict the Trust Evaluator is (Section 6).

---

## 5. Trust Evaluator — Signal Design

**Principle:** never trust the local model's self-reported confidence. Combine multiple independent, harder-to-fake signals and require agreement across signals before skipping escalation.

### 5.1 Signals used

| # | Signal | How it works | Cost | Reliability notes |
|---|---|---|---|---|
| 1 | **Self-consistency (sampling agreement)** | Run the local model N times (e.g., N=3–5) at temperature > 0 on the same query. Compare answers for agreement (exact match for short/structured answers; embedding similarity for free text). | Free (local tokens) | Strong signal; catches cases where the model's "knowledge" is unstable rather than just its phrasing. Primary signal given the rules make this free. |
| 2 | **Token-level entropy / logprobs** | Inspect the actual probability distribution during generation (not the model's stated confidence). High entropy / low top-token probability → treat as less trustworthy. | Free (already computed during generation) | Correlates with linguistic uncertainty, not always factual correctness — used as a secondary filter, not sole gate. |
| 3 | **External/structural verification** | Where possible: execute code and check for errors, validate output format (JSON parses, numeric ranges plausible), or cross-check against any retrieved/structured context. | Free/cheap | Most trustworthy signal when applicable, but not all tasks support it — depends on task type revealed at kickoff. |
| 4 | **Category-based historical reliability** | Track, per task category, how often local answers (that were later checked, either via escalation or eval) turned out correct. Feed into the Threshold Adjuster (Section 6). | Free (bookkeeping only) | Improves over the course of the task set; weak/noisy early on with few samples per category. |

### 5.2 Decision rule (v1 — simple, safe default)

```
escalate = (
    self_consistency_score < consistency_threshold
    OR entropy_score > entropy_threshold
    OR (structural_check_available AND structural_check_failed)
)
```

Start with a simple OR-based rule (any red flag triggers escalation) to bias toward accuracy in early testing, then tune thresholds against the local eval set to trade off toward fewer escalations without dropping below the accuracy floor.

### 5.3 Decision rule (v2 — stretch goal, budget-aware)

Same signals, but thresholds are not fixed constants — they are adjusted dynamically by the Threshold Adjuster (Section 6). Only build this after v1 is working and tested; do not ship v2 unless it demonstrably beats v1 on the local eval harness.

---

## 6. Budget-Aware Threshold Adjuster (Stretch Goal)

**Principle:** routing decisions are not independent — they share a global token budget and a global accuracy floor across the whole task set. Later decisions should be informed by what's been learned so far.

### 6.1 State tracked
- Remote tokens spent so far / estimated tasks remaining
- Running accuracy estimate (via verification signals, not ground truth, since ground truth is generally unavailable at inference time)
- Per-category counters: local-trusted-and-correct, local-trusted-and-wrong (if discoverable), escalated

### 6.2 Adjustment logic (bandit-style, no training loop)
- Maintain a simple success/failure counter per task category (e.g., Beta distribution parameters, or just a moving average).
- If budget pressure is high (spending faster than task count justifies) → tighten thresholds (escalate less).
- If a category has a strong local-success track record → raise its escalation bar (trust local more for that category).
- If a category has a poor track record → lower its escalation bar (escalate more readily).

### 6.3 Explicit risk
With a small/bounded task set (likely the case here), there may not be enough trials per category for the adaptive layer to calibrate meaningfully. **Mitigation:** always keep v1 (static thresholds) as the shipped fallback; only enable v2 if A/B testing on the local eval set shows a clear improvement.

---

## 7. Local Model Selection

- Must run within the standardized scoring environment's hardware constraints (confirm exact limits at kickoff — do not assume dev-machine specs carry over).
- Favor a small, well-quantized instruction-tuned model (e.g., in the 0.5B–3B range depending on confirmed constraints) served via a lightweight local runtime (e.g., llama.cpp / Ollama).
- Model choice should be finalized on Day 1 after kickoff reveals the standardized environment specs and task types.

---

## 8. Architecture / Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Local inference runtime | llama.cpp / Ollama | Lightweight, quantization support, works within constrained hardware |
| Remote inference | Fireworks AI API | Required by competition rules |
| Orchestration | Python | Fast to iterate in 4 days |
| Trust Evaluator | Custom Python module (no ML framework needed for v1) | Signals are statistical/rule-based, not learned |
| Logging / eval harness | Local JSON/CSV logs + a small eval script | Needed to tune thresholds before submission |

---

## 9. Evaluation Plan

Per competition guidance, a local eval step is required before submission.

1. Build a held-out mini task set (self-constructed, mimicking expected task categories) to test the router before real tasks are revealed.
2. Track for each run: total remote tokens used, accuracy (via whatever proxy is available — exact match, rubric, or self-built judge), and escalation rate per category.
3. Tune `consistency_threshold` and `entropy_threshold` against this harness to find the lowest token spend that keeps accuracy above a safety margin (not just the bare minimum threshold, to leave room for task variance).
4. Only after v1 is validated, test v2 (budget-aware) head-to-head against v1 on the same harness before deciding whether to ship it.

---

## 10. Day-by-Day Plan (4 Days)

| Day | Focus |
|---|---|
| **Day 1** | Kickoff: confirm standardized environment specs, task types, models available. Set up local model runtime + Fireworks API access. Build Local Executor + basic logging. |
| **Day 2** | Build Trust Evaluator v1 (self-consistency + entropy + structural checks where applicable). Build eval harness with a self-constructed mini task set. Start threshold tuning. |
| **Day 3** | Finish threshold tuning against eval harness. If time allows, build Threshold Adjuster v2 and A/B test against v1. Freeze whichever version wins. Stress-test edge cases (empty responses, malformed structured tasks, API failures/timeouts). |
| **Day 4** | Final integration test on full pipeline. Add fallback/error handling (e.g., if Fireworks call fails, retry or gracefully degrade). Write submission documentation. Buffer for unexpected issues. |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Local model too weak within standardized hardware constraints, forcing over-escalation | Confirm hardware specs immediately at kickoff; pick smallest viable model; test early on Day 1. |
| Self-consistency sampling costs too much local compute/time (even if tokens are free) | Cap N (samples) low (3) by default; only increase for borderline cases. |
| Structural/execution verification not applicable to revealed task types | Build it as a pluggable, optional signal — system must work with just signals 1 and 2 if needed. |
| v2 (budget-aware) destabilizes routing on a small task set | Always keep v1 as the default; only switch to v2 if validated via A/B test on eval harness. |
| Accuracy threshold unknown until kickoff | Build with a safety margin during tuning rather than optimizing to the bare minimum. |

---

## 12. Open Questions (resolve at kickoff)

- What are the exact standardized environment hardware specs (CPU/GPU/RAM) for final scoring?
- What local and remote models are available/permitted?
- How is accuracy scored — exact match, rubric, or LLM-judge?
- What is the accuracy threshold floor?
- Are tasks revealed all at once, or streamed one at a time (affects whether budget-aware adjustment is even meaningful)?
