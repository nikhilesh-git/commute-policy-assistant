"""Choose the relevance gate (settings.min_similarity) from data. Offline, no LLM.

For every eval case it records the best embedding similarity of the question to
the asking company's active policy, then shows, per candidate threshold:
  lost_answers     answerable questions the gate would wrongly turn into "not covered"
  gated_unanswered unanswerable questions stopped before the LLM (a saved call,
                   and a guaranteed honest reply)
Questions naming another company are excluded: they are refused before retrieval.

Usage: python -m evaluation.threshold_sweep
"""

import logging
import os
import tempfile
from pathlib import Path

from evaluation.run_eval import DEFAULT_CASES, NIMBUS_V2, load_cases

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def collect() -> list[tuple[dict, float]]:
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="sweep-")
    from app.answer import other_company_aliases
    from app.config import Settings
    from app.db import connect, init_db
    from app.guards import mentioned_company
    from app.ingestion import ingest_policy_version, parse_title
    from app.retrieval import retrieve
    from app.seed import seed, seed_policies
    from app.vectorstore import VectorStore

    logging.disable(logging.WARNING)
    settings = Settings(data_dir=Path(os.environ["DATA_DIR"]), _env_file=None)
    init_db(settings.sqlite_path)
    conn = connect(settings.sqlite_path)
    seed(conn)
    store = VectorStore(settings.chroma_path)
    seed_policies(conn, store, settings)

    out = []
    cases = load_cases(DEFAULT_CASES)
    for stage in ("initial", "after_nimbus_v2"):
        if stage == "after_nimbus_v2":
            content = NIMBUS_V2.read_text(encoding="utf-8")
            ingest_policy_version(conn, store, settings, "nimbus", "transport-policy", parse_title(content), content)
        for case in (c for c in cases if c["input"].get("stage", "initial") == stage):
            company, question = case["input"]["company"], case["input"]["question"]
            if mentioned_company(question, other_company_aliases(conn, company)):
                continue
            out.append((case, retrieve(conn, store, settings, company, question).top_similarity or 0.0))
    return out


def main() -> None:
    data = collect()

    def statuses(case):
        s = case["expected"]["status"]
        return s if isinstance(s, list) else [s]

    answerable = sorted((sim, c["id"]) for c, sim in data if statuses(c) == ["answered"])
    unanswerable = sorted((sim, c["id"]) for c, sim in data if statuses(c) == ["not_covered"])

    print(f"answerable ({len(answerable)}): lowest top-similarities")
    for sim, cid in answerable[:5]:
        print(f"   {sim:.3f}  {cid}")
    print(f"unanswerable ({len(unanswerable)}): all top-similarities")
    for sim, cid in unanswerable:
        print(f"   {sim:.3f}  {cid}")

    print("\nthreshold | lost_answers | gated_unanswered")
    for t in THRESHOLDS:
        lost = sum(sim < t for sim, _ in answerable)
        gated = sum(sim < t for sim, _ in unanswerable)
        print(f"   {t:.2f}   |      {lost:2d}      |     {gated}/{len(unanswerable)}")


if __name__ == "__main__":
    main()
