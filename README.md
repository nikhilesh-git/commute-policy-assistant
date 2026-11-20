# Commute Policy Assistant

Employees ask transport questions ("Can I book a cab on Saturday?"). The assistant
answers **only from their own company's current policy**, cites the document,
version and section, says plainly when the policy is silent, and never answers
from another company's document, whatever the question or the retrieved text
tries to make it do.

Python · FastAPI · Chroma (embedded) · local MiniLM embeddings · BM25 · Groq (free tier) · JWT · SQLite

## Run it

```bash
docker compose up --build
```

That is the whole setup. Open <http://localhost:8000/docs>. With no `.env` the app
uses an offline "extractive" fake LLM and a random JWT secret, so it starts and
the tests and evaluation run without any key. (Port 8000 taken? `API_PORT=8080 docker compose up --build`.)

For real answers, create `.env` from `.env.example` and set:

```
LLM_PROVIDER=groq
GROQ_API_KEY=<your free Groq key>
```

### Tests and evaluation

```bash
docker compose run --rm api pytest                                   # 238 tests, ~90 s, no key, no network
docker compose run --rm api python -m evaluation.run_eval            # 40-case eval, offline, ~5 s
docker compose run --rm api python -m evaluation.run_eval --llm groq --repeat 2   # real model, ~10 min
```

Results and the full report: [EVALUATION.md](EVALUATION.md).

**Evaluation set (submission file):** [`evaluation/results/groq-hybrid.jsonl`](evaluation/results/groq-hybrid.jsonl):
40 cases, one per line, with the fields `id, input, expected, actual, pass, notes`, from the
real model with the shipped configuration. (`evaluation/cases.jsonl` holds the same cases
without results; it is the input the runner reads.)

### Try it

Demo users (local demo database only), password `demo1234`:

| Company | Employee | Admin |
|---|---|---|
| Nimbus Tech | `alice@nimbus` | `admin@nimbus` |
| Orion Finserv | `bob@orion` | `admin@orion` |
| Kestrel Health | `chitra@kestrel` | `admin@kestrel` |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H "Content-Type: application/json" \
  -d '{"username":"alice@nimbus","password":"demo1234"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -s -X POST localhost:8000/ask -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"Will a guard come with me on a late night drop?"}'
```

```json
{"status": "answered",
 "answer": "Yes, a security escort is mandatory on every drop of a woman employee after 8:00 PM, ...",
 "citations": [{"document": "Nimbus Tech Employee Transport Policy", "version": 1,
                "section": "4.1 Escort for women employees", "chunk_id": "2:6",
                "quote": "A security escort (guard) is mandatory on every drop of a woman employee after 8:00 PM ..."}],
 "request_id": "..."}
```

Publish a new policy version through the admin API (the company comes from the admin's token):

```bash
docker compose exec api python scripts/upload_policy.py policies/nimbus/transport-policy.v2.md --user admin@nimbus
```

`python scripts/demo.py` (stdlib only) walks through every scenario: answers,
cross-company traps, "not covered", injection attempts and a live version update.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/token` | none | Mock JWT issuer: `{username, password}` → token with `company_id` and `role` |
| POST | `/ask` | employee/admin | `{question}` → `answered` / `not_covered` (200) or `try_again` (503) |
| POST | `/admin/policies/{slug}/versions` | admin | Ingest a new version of a policy (Markdown) |
| GET | `/admin/policies` | admin | Versions and status, own company only |
| GET | `/health` | none | DB, vector store, LLM provider, active policies |

No request body or path carries a company id. Request bodies forbid unknown
fields, so sending `company_id` is a 422, not a silent override.

## How it works

```
POST /ask ── JWT verified (HS256, iss/aud/exp, company must exist) ── company_id
   │
   ├─ question names another company?  ─────────────────────► "only your own policy" (0 LLM calls)
   ├─ retrieve(company_id, question)
   │     Chroma collection policy_<company_id>, filter version_id ∈ active versions (from SQLite)
   │     + BM25 over the same company's active chunks, merged with Reciprocal Rank Fusion
   │     + post-check: every candidate must match company + active version, or it is dropped
   │     store down / > 1 s ────────────────────────────────────► "try again" (503)
   │     best similarity < 0.25 ────────────────────────────────► "not covered" (0 LLM calls)
   ├─ prompt: top 3 chunks as <excerpt id="C1..C3">, question in <question>, all escaped
   ├─ LLM (Groq qwen3.8-27b, JSON mode), max 2 calls, 2.7 s deadline
   │     timeout / 5xx / malformed JSON + ≥1 s left → one retry; 429 → no retry
   │     nothing usable ─────────────────────────────────────────► "try again" (503)
   ├─ validate: cited labels ⊆ sent; quote found in a cited chunk; no other company named
   │     any failure ───────────────────────────────────────────► "not covered"
   └─ citations built from chunk metadata (document, version, section), logged to query_log
```

| Module | Responsibility |
|---|---|
| `app/config.py` | Every limit and threshold, overridable by env var |
| `app/auth.py` | Mock issuer, JWT verification, role checks |
| `app/ingestion.py` | Markdown → sections → chunks; atomic version switch |
| `app/vectorstore.py` | Chroma: one collection per company |
| `app/keyword_search.py` | BM25 (hybrid retrieval) |
| `app/retrieval.py` | Company + active-version search, fusion, post-check, relevance gate |
| `app/prompts.py` | Prompt text and escaping |
| `app/llm.py` | `LLMClient` interface, Groq client, FakeLLM, per-question call budget |
| `app/guards.py` | Input checks, injection sanitiser, output validation |
| `app/answer.py` | The pipeline above, orchestration only |
| `app/routers/` | HTTP only; no business logic |

## Design decisions

### Company isolation is enforced on the server, in layers
1. The company comes **only** from a verified JWT. Bodies forbid extra fields.
2. Each company has its **own Chroma collection**, named from the token's company,
   so a forgotten filter still cannot return another company's chunks.
3. Search is filtered to the **active version ids**, read from SQLite.
4. A **post-check** re-verifies every candidate against both its Chroma metadata and
   its SQLite row; mismatches are dropped and logged as `SECURITY:`. Tests plant
   foreign chunks to prove it fires.
5. Questions that name another company are refused before retrieval; answers that
   name one are rejected after generation.

The LLM is never asked to "only use company X": by the time it runs, nothing from
another company is in front of it.

### Policy updates are atomic
SQLite is the source of truth; Chroma is only an index, and it has no transactions.
So a new version's vectors are written first (tagged with the new `version_id`,
invisible because retrieval only searches active versions), counted, and then **one
SQLite transaction** inserts the chunk rows, marks the old version `superseded` and
the new one `active`. A partial unique index makes two active versions impossible.
Any failure deletes the new vectors and marks the version `failed`; the old version
keeps answering. A crash mid-ingestion is cleaned up at the next startup.
Tests force failures halfway through the vector writes and inside the final
transaction (via a SQLite trigger) and check nothing half-updated is visible.

### How it stays within 2 LLM calls and 3 cited chunks
- **Calls:** normally **one** call per question. `LLMBudget` counts calls per
  question and raises on a third; the only second call is a retry after a
  timeout, provider error or malformed JSON, and only if at least 1 s of the 2.7 s
  deadline is left. Refusals, off-topic questions and store failures make **zero**
  calls. Every failure-path test asserts the call count, and the eval reads it from
  the query log: max 1, mean 0.85 over 80 real requests.
- **Chunks:** retrieval returns at most `MAX_CITATIONS = 3` chunks and only those
  are in the prompt, so the model *cannot* cite a fourth. The validator also drops
  any label that was not sent.

### The LLM only chooses words
Authorisation, isolation, versions, limits, refusals and validation are Python.
The model picks which of the given excerpts support its answer (by label) and
writes a short answer plus one quoted sentence. The server checks the quote is
really in a cited excerpt and builds the citation from stored metadata, so a
citation can always be checked against the document.

### Prompt injection: several independent layers
- **In documents:** instruction-like sentences are replaced at ingestion
  (`[removed: instruction-like text]`) and the admin gets a warning.
- **In the prompt:** excerpts and question are wrapped as data and escaped, so
  `</question>` in a question cannot close our tag.
- **After the model:** the quote, citation and company-name checks.
- **By design:** the model has no tools and no other company's data, so the worst
  a successful injection can do is a bad answer for the attacker's own company,
  and validation usually turns that into "not covered".

The sample policies contain one obvious injection (caught by the sanitiser) and one
subtle one that deliberately slips past the regexes, to test the remaining layers.

### Failures give a clear reply, never a hang or half answer
Vector store down or slow (> 1 s): `try_again`. LLM timeout, 5xx, malformed output:
one retry if time allows, then `try_again`. Rate limit: no retry. An unexpected
exception in the provider client: `try_again`, logged. Unverifiable answer:
`not_covered`, pointing to the transport desk.

### Why no LangChain/LlamaIndex
The things being evaluated (tenant filters, version filters, call budget, citation
validation) would sit inside a framework. Plain modules keep each control in one
visible function.

## Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| Strict output validation (quote must be in a cited chunk) | No unverifiable answers | A model that paraphrases instead of quoting loses answers |
| Only 3 chunks to the model | "≤ 3 citations" holds by construction; short prompts | Questions needing 4+ sections can get an incomplete answer |
| Hybrid retrieval (BM25 + embeddings) | +1 case, hit@1 0.83 → 0.90, +2-6 ms | One regression; a second ranking to explain |
| Relevance gate at 0.25, not higher | No answerable question wrongly refused | Half the unanswerable questions still cost an LLM call (which then says "not covered") |
| Refuse any question naming another company | Simple, leak-proof | A future company named like a common word would cause false refusals |
| Embedded Chroma + SQLite, single process | One command, nothing to operate | One instance only (startup cleanup assumes no concurrent ingester) |
| Old vectors deleted at next startup, not at the switch | In-flight requests never see a vector vanish | Superseded vectors use disk until a restart (never searched) |
| Deterministic string scoring in the eval | Same score every run, no quota | Unexpected but correct phrasings can fail a fact check |

## Results (summary)

40-case evaluation, `qwen/qwen3.8-27b` on Groq, 80 requests per run. Full detail in [EVALUATION.md](EVALUATION.md).

| | Dense (baseline) | Hybrid (shipped) |
|---|---|---|
| Passed | 38/40 | **39/40** |
| Cross-company leaks / injections obeyed | 0 / 0 | 0 / 0 |
| p95 latency (LLM included) | 888 ms | 985 ms |
| Max LLM calls per question | 1 | 1 |

## Things found along the way

- Chroma's `DefaultEmbeddingFunction` reloads the ONNX model on every call and pads
  to 256 tokens: retrieval took ~200 ms. A 10-line subclass (`UnpaddedMiniLM`)
  brought it to ~10 ms with identical vectors (tested).
- With Chroma's default `sync_threshold`, small indexes live only in memory; on
  Windows the index cache holds ~100, and an evicted one becomes unreadable
  ("Nothing found on disk"). `sync_threshold=1` fixes it (regression test included).
- `llama-3.1-8b-instant` had been retired by Groq; the model was chosen by
  benchmarking the available ones on our own prompts.

## Assumptions

- Policies are Markdown with numbered `##`/`###` headings; DOCX/PDF parsing is out of scope.
- The assistant explains the applicable rule with a citation; it does not decide
  personal eligibility (that would need HR data and would put a business decision
  in the model's hands).
- English questions; one policy document per company (the schema allows more).

## Project layout

```
app/                FastAPI service (modules above), routers/ for HTTP
policies/           3 fictional companies + Nimbus v2; contradiction table in policies/README.md
evaluation/         cases.jsonl, scoring, metrics, runner, threshold sweep, results/
scripts/            upload_policy.py, demo.py
tests/              pytest suite (fake LLM, temporary storage, mocked HTTP for Groq)
```
