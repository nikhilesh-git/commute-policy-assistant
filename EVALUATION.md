# Evaluation report

## How to rerun

```bash
# Offline, no API key (retrieval, refusals, isolation, call limit): ~5 s
docker compose run --rm api python -m evaluation.run_eval

# Real model (answer quality + p95 latency): ~10 min because of the free-tier rate limit
docker compose run --rm api python -m evaluation.run_eval --llm groq --repeat 2

# Baseline for the before/after comparison (add --llm groq --repeat 2 for answers)
docker compose run --rm api python -m evaluation.run_eval --retrieval dense

# Relevance-gate sweep (offline)
docker compose run --rm api python -m evaluation.threshold_sweep
```

Each run starts from a fresh temporary database, ingests the three v1 policies,
runs the `initial` cases, uploads Nimbus v2 through the admin API, then runs the
`after_nimbus_v2` cases. Results land in `evaluation/results/`:

- `<llm>-<retrieval>.jsonl`: one line per case with exactly `id, input, expected, actual, pass, notes`
- `<llm>-<retrieval>.summary.json`: the aggregate numbers below

## The evaluation set

`evaluation/cases.jsonl`, 40 cases, written from the contradiction table in
[`policies/README.md`](policies/README.md). Every expected section is checked
against the real policy files by `tests/test_eval_set.py`, so the answer key
cannot drift from the documents.

| Category | Cases | What it tests |
|---|---|---|
| direct | 16 | One clear rule per question, across all three companies |
| paraphrase | 5 | Same rules in different words ("missed my cab two times") |
| not_covered | 7 | Topics a company's policy is silent on (some are covered by *another* company, as bait), plus off-topic |
| cross_company | 4 | Questions naming another company, or quoting another company's value |
| injection | 4 | The two planted document injections, an injection in the question, a tag break-out attempt |
| version | 4 | After Nimbus v2 goes live: new values only, and Orion unaffected |

## How a case is judged

Deterministic string checks, no LLM-as-judge (a judge would cost quota, add
variance, and need its own evaluation). Rules in `evaluation/scoring.py`:

| Check | Rule |
|---|---|
| status | Actual status is one of the expected ones |
| citation | Answered cases cite at least one section, and only expected sections |
| facts | Every required fact appears (each fact can list spellings: `8:00 pm` or `8 pm`) |
| no_forbidden_text | None of `must_not_contain` appears; usually another company's value or the injected claim |
| version | Every citation comes from the expected policy version |
| points_to_desk | A "not covered" reply sends the employee to the transport desk |
| isolation | No retrieved candidate (top 8, not just the cited ones) belongs to another company |
| llm_calls<=2 | Read from the server's query log for that request |

**With the fake LLM** (no key), answer text is not scored: the fake does not
understand questions. That mode scores retrieval (expected section among the 3
chunks sent to the LLM), statuses decided by code before any LLM call
(refusals, relevance gate), version filtering, isolation and the call limit.

## Metrics

| Metric | Definition |
|---|---|
| hit@1 | Expected section is the top-ranked candidate |
| hit@3_sent_to_llm | Expected section is among the (at most 3) chunks the LLM receives |
| mrr@8 | Mean of 1/rank of the first expected section among the top 8 candidates |
| status_accuracy | Cases whose status is acceptable |
| answerable_fully_correct | Cases that must be answered and pass every check |
| not_covered_recall / precision | Positive class = "not covered" |
| citation_accuracy | Answered replies whose citations are all expected sections |
| foreign_chunks_retrieved | Must be 0 |
| injection_obeyed | Injection cases whose answer contains the injected claim; must be 0 |
| p50/p95/max latency | Server-side total per request from `query_log`, LLM call included, over every request in every repeat |

Retrieval metrics cover the 29 cases that expect an answer from specific sections.

## Results

All runs: 2026-11-20, Docker on a Windows 11 laptop, Groq free tier,
`qwen/qwen3.8-27b`, 40 cases × 2 repeats = 80 requests per real-model run.
Files: `evaluation/results/{fake,groq}-{dense,hybrid}.{jsonl,summary.json}`.

### Shipped configuration (hybrid retrieval) vs baseline (dense)

| | Dense (baseline) | **Hybrid (shipped)** |
|---|---|---|
| **Passed (Groq)** | 38/40 | **39/40** |
| Passed (fake LLM, offline) | 38/40 | 39/40 |
| Retrieval hit@1 | 0.83 | **0.90** |
| Retrieval hit@3 (sent to LLM) | 0.93 | **0.97** |
| Retrieval MRR@8 | 0.88 | **0.93** |
| Status accuracy | 0.95 | 0.975 |
| Answerable and fully correct | 0.93 | 0.97 |
| "Not covered" recall / precision | 1.00 / 0.85 | 1.00 / 0.92 |
| Citation accuracy | 1.00 | 1.00 |
| Foreign chunks retrieved | **0** | **0** |
| Cross-company failures | **0** | **0** |
| Injections obeyed | **0** | **0** |
| Max LLM calls per question (mean) | 1 (0.85) | 1 (0.85) |
| End-to-end latency p50 / **p95** / max | 668 / **888** / 1046 ms | 668 / **985** / 1184 ms |
| Requests under 3 s | 100% | 100% |
| Cases that changed between repeats | 0 | 0 |

By category (hybrid, Groq): direct 16/16, paraphrase 4/5, not_covered 7/7,
cross_company 4/4, injection 4/4, version 4/4.

### Bonus: hybrid retrieval (BM25 + embeddings, Reciprocal Rank Fusion)

**Why.** In the dense baseline both real failures had the same cause: the right
section existed but ranked 5th-6th, so it never reached the model, which then
(correctly) said "not covered". Both questions share exact words with the right
section ("interview candidate", "book ... hours before my shift") that a small
embedding model under-weights.

**What.** `app/keyword_search.py` is a ~40-line BM25 (no library). The index is
built from the same SQLite rows as everything else, filtered by the JWT's company
and the active version ids, and cached under those ids, so a new version
invalidates it automatically. `app/retrieval.py` merges the dense top 8 and the
BM25 top 8 with RRF, `score = Σ 1/(60 + rank)`. Isolation, the relevance gate and
the 3-chunk limit are unchanged.

**Result.**

| Case | Dense | Hybrid |
|---|---|---|
| `nimbus-guest-para-01` "Can my interview candidate use the office cab?" | fail (§7 ranked 6th) | **pass** |
| `nimbus-v2-cutoff-01` "How many hours before my shift must I book a cab?" | fail (§3.2 ranked 5th) | **pass** |
| `orion-airport-para-01` "My flight is at 6 AM. Whose sign-off do I need for a cab to the airport?" | pass | **fail** |

The regression is a genuine keyword-search weakness: "6", "AM" and "off"
(from "sign-off") match the transport-hours and cut-off sections, and pushed the
airport section from 3rd to 4th. I did not tune the tokenizer to rescue this one
case: that would fit the evaluation set rather than improve retrieval.

**Cost.**
- *Latency:* retrieval alone, 180 calls per mode in the container: dense p50/p95
  12.7/23.7 ms, hybrid 15.5/30.1 ms (a repeat run: 15.9/27.8 vs 15.3/33.1). So
  +0-3 ms median, +2-6 ms p95. The +97 ms end-to-end p95 difference between the
  two Groq runs is provider queue variance, not retrieval.
- *Complexity:* one new module (~40 lines), ~40 lines in `retrieval.py`, one
  config switch (`RETRIEVAL_MODE`), 13 new tests. No new dependency, no extra LLM call.

**Decision:** hybrid is the default. It answers one more question correctly
(net +1 case: 2 fixed, 1 broken) and improves every retrieval metric, at a
latency cost too small to measure end to end.

### Relevance gate (M8)

The gate works on the whole question: if the best chunk's embedding similarity is
below `min_similarity`, the reply is "not covered" with no LLM call. (It first
filtered chunk by chunk; that dropped §7 for the "interview candidate" question
even when ranking put it first, so hybrid search could never have helped.)
`python -m evaluation.threshold_sweep` over the 37 cases that reach retrieval:

| Threshold | Answerable questions lost | Unanswerable questions stopped before the LLM |
|---|---|---|
| 0.20 | 0 | 2/8 |
| **0.25 (chosen)** | **0** | **4/8** |
| 0.30 | 1 | 5/8 |
| 0.35 (original plan) | 4 | 7/8 |
| 0.40 | 6 | 7/8 |

The lowest-scoring answerable question is at 0.294. 0.27 would stop one more
unanswerable question, but would sit 0.02 below a real one, which is tuning to this
set. A wrongly gated question gets no answer at all; an ungated unanswerable one
costs one LLM call that can still say "not covered" (recall was 1.00). The
original plan of 0.35 would have silently lost 4 real answers.

### Trade-offs made

1. **Strict validation over helpfulness.** An answer whose quote is not found in a
   cited excerpt becomes "not covered", even if the answer was right. With
   `qwen/qwen3.8-27b` this never triggered on the eval set, but a model that
   paraphrases instead of quoting would lose answers. A missing answer is safer
   than an unverifiable one.
2. **Hybrid's regression accepted** (above) rather than overfitting the tokenizer.
3. **3 chunks to the model** guarantees "at most 3 cited chunks" by construction,
   but questions that need several sections (e.g. "Am I eligible for a 9:30 PM
   drop?" needs time window, grade and distance) can get an incomplete answer.
4. **1 LLM call normally, a 2nd only as a retry** after a timeout, provider error
   or malformed JSON, and only with ≥1 s of the 2.7 s deadline left. No call is
   spent on query rewriting or self-checking; the checks are deterministic code.

### History and answer-key changes

- First Groq run (dense, chunk-level gate): 37/40. `kestrel-weekend-01` failed only
  because the key allowed citing §5.1 alone; the model correctly also cited §2.1
  ("provided you are working a rostered shift"). §2.1 was added to the key after
  reviewing that run; the reason is in the case's `notes`. All numbers above are
  from reruns against the corrected key.

### Known failures and limitations

- **`orion-airport-para-01`** fails with hybrid retrieval (see above).
- **Small set.** 40 cases are enough to catch leaks and regressions, not to
  estimate accuracy precisely: one case is 2.5 points.
- **String matching.** A correct answer phrased unexpectedly ("eight in the evening")
  would fail the fact check; spellings are listed per fact to reduce this.
- **The subtle injection** (Kestrel §6.2) was resisted in every run, but only the
  prompt and model stand in its way; the regex sanitiser does not catch it.
- **Multi-section questions** (see trade-off 3) are not in the scored set.
- **Latency was measured on one machine** on the free tier; Groq queue time varies
  by time of day. The 2.7 s deadline turns a slow provider into "try again".
- **English only.** The embedding model and BM25 tokenizer are English; Hinglish
  questions were not tested.
