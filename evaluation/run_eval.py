"""Run the evaluation set in-process and write results JSONL + a summary.

Usage:
  python -m evaluation.run_eval                                # fake LLM, no key needed
  python -m evaluation.run_eval --llm groq --repeat 2          # real model, answer quality + p95
  python -m evaluation.run_eval --llm groq --retrieval dense   # the baseline, for before/after

Flow: fresh temporary database -> seed all v1 policies -> run 'initial' cases
-> upload Nimbus v2 through the admin API -> run 'after_nimbus_v2' cases.

Outputs (stable names, so before/after runs can be committed and compared):
  evaluation/results/<llm>-<retrieval>.jsonl          one line per case: id, input, expected, actual, pass, notes
  evaluation/results/<llm>-<retrieval>.summary.json   aggregate metrics (see EVALUATION.md)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_CASES = HERE / "cases.jsonl"
RESULTS_DIR = HERE / "results"
STAGES = ["initial", "after_nimbus_v2"]
EMPLOYEE_FOR_COMPANY = {"nimbus": "alice@nimbus", "orion": "bob@orion", "kestrel": "chitra@kestrel"}
NIMBUS_V2 = Path("policies/nimbus/transport-policy.v2.md")


def load_cases(path: Path) -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "input", "expected"):
            if key not in case:
                raise ValueError(f"{path}:{n} missing field {key!r}")
        if case["input"].get("stage", "initial") not in STAGES:
            raise ValueError(f"{path}:{n} unknown stage {case['input']['stage']!r}")
        cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")
    return cases


class Runner:
    def __init__(self, client, delay_s: float):
        from app.config import get_settings
        from app.db import connect
        from app.seed import DEMO_PASSWORD

        self.client = client
        self.delay_s = delay_s
        self.settings = get_settings()
        self.store = client.app.state.vectorstore
        self.conn = connect(self.settings.sqlite_path)
        self.password = DEMO_PASSWORD
        self.tokens = {c: self._login(u) for c, u in EMPLOYEE_FOR_COMPANY.items()}

    def _login(self, username: str) -> dict:
        r = self.client.post("/auth/token", json={"username": username, "password": self.password})
        r.raise_for_status()
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    def upload_nimbus_v2(self) -> None:
        from app.ingestion import parse_title

        content = NIMBUS_V2.read_text(encoding="utf-8")
        r = self.client.post("/admin/policies/transport-policy/versions",
                             json={"title": parse_title(content), "content": content},
                             headers=self._login("admin@nimbus"))
        if r.status_code != 201:
            raise RuntimeError(f"uploading Nimbus v2 failed: {r.status_code} {r.text}")

    def run_case(self, case: dict) -> dict:
        from app.retrieval import retrieve

        company, question = case["input"]["company"], case["input"]["question"]
        r = self.client.post("/ask", json={"question": question}, headers=self.tokens[company])
        body = r.json()
        log = self.conn.execute("SELECT * FROM query_log WHERE request_id = ?",
                                (body.get("request_id"),)).fetchone()

        # Retrieval diagnostics: the same function /ask uses, called again so we can
        # see every candidate (the API only returns what was cited).
        diag = retrieve(self.conn, self.store, self.settings, company, question)
        citations = body.get("citations", [])
        actual = {
            "http_status": r.status_code,
            "status": body.get("status"),
            "answer": body.get("answer", body.get("detail", "")),
            "cited_sections": [c["section"].split(" ", 1)[0] for c in citations],
            "cited_versions": sorted({c["version"] for c in citations}),
            "sent_sections": [c.section_id for c in diag.chunks],
            "sent_versions": sorted({c.version_no for c in diag.chunks}),
            "candidate_sections": [c.section_id for c in diag.candidates],
            "top_similarity": round(diag.top_similarity, 3) if diag.top_similarity is not None else None,
            # Another company's chunk among the candidates, or one the post-check had to
            # drop (caught, but still a filter failure). Must be 0 for every case.
            "foreign_chunks": sum(c.company_id != company for c in diag.candidates) + diag.dropped,
            "llm_calls": log["llm_calls"] if log else 0,
            "latency_ms": round(log["total_ms"], 1) if log else None,
            "llm_ms": round(log["llm_ms"], 1) if log and log["llm_ms"] else 0.0,
        }
        if actual["llm_calls"] and self.delay_s:
            time.sleep(self.delay_s)  # stay under the provider's tokens-per-minute limit
        return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--llm", choices=["fake", "groq"], default="fake")
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="hybrid",
                        help="hybrid is what the app ships with; dense is the baseline")
    parser.add_argument("--repeat", type=int, default=1, help="run every case N times (more latency samples)")
    parser.add_argument("--delay", type=float, default=None,
                        help="seconds to pause after each LLM-using request (default: 8 for groq, 0 for fake)")
    args = parser.parse_args()
    delay = args.delay if args.delay is not None else (8.0 if args.llm == "groq" else 0.0)

    # Isolated, throwaway data so an eval run never touches the real database.
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="eval-")
    os.environ["LLM_PROVIDER"] = args.llm
    os.environ["RETRIEVAL_MODE"] = args.retrieval
    os.environ["SEED_POLICIES_ON_STARTUP"] = "true"

    from fastapi.testclient import TestClient

    from app.main import create_app
    from evaluation.metrics import summarise
    from evaluation.scoring import score_case

    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("app", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    mode = "llm" if args.llm != "fake" else "fake"
    cases = load_cases(args.cases)
    started = time.perf_counter()
    first_pass: dict[str, dict] = {}
    pass_history: dict[str, set[bool]] = {}
    all_requests: list[dict] = []

    with TestClient(create_app()) as client:
        if client.get("/health").json().get("active_policies") != 3:
            raise RuntimeError("policies were not seeded; see the log above")
        runner = Runner(client, delay)
        for stage in STAGES:
            if stage == "after_nimbus_v2":
                runner.upload_nimbus_v2()
            stage_cases = [c for c in cases if c["input"].get("stage", "initial") == stage]
            for rep in range(args.repeat):
                for case in stage_cases:
                    actual = runner.run_case(case)
                    passed, checks, notes = score_case(case["expected"], actual, mode)
                    all_requests.append(actual)
                    pass_history.setdefault(case["id"], set()).add(passed)
                    if rep == 0:
                        first_pass[case["id"]] = {
                            "id": case["id"], "input": case["input"], "expected": case["expected"],
                            "actual": actual, "pass": passed,
                            "notes": "; ".join(notes) or case.get("notes", ""),
                            "checks": checks,
                        }
                    mark = "PASS" if passed else "FAIL"
                    print(f"  [{mark}] {case['id']:28} {actual['status'] or '-':11} "
                          f"{actual['latency_ms'] or 0:6.0f} ms  {'; '.join(notes)[:90]}", flush=True)

    rows = [first_pass[c["id"]] for c in cases]
    unstable = [cid for cid, seen in pass_history.items() if len(seen) > 1]
    summary = summarise(rows, all_requests, mode, unstable)
    summary = {
        "run": {"llm": args.llm, "model": os.environ.get("GROQ_MODEL") if args.llm == "groq" else "fake",
                "retrieval": args.retrieval, "repeat": args.repeat, "cases": len(cases),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "wall_clock_s": round(time.perf_counter() - started)},
        **summary,
    }

    lines = "".join(
        json.dumps({k: row[k] for k in ("id", "input", "expected", "actual", "pass", "notes")},  # required fields
                   ensure_ascii=False) + "\n"
        for row in rows
    )
    name = f"{args.llm}-{args.retrieval}"
    try:
        out_dir = RESULTS_DIR
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{name}.jsonl").write_text(lines, encoding="utf-8")
    except PermissionError:
        # e.g. a bind-mounted folder owned by a different host user: keep the results anyway.
        out_dir = Path(tempfile.mkdtemp(prefix="eval-results-"))
        (out_dir / f"{name}.jsonl").write_text(lines, encoding="utf-8")
        print(f"\n{RESULTS_DIR} is not writable; results written to {out_dir} instead")
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / name}.jsonl and {out_dir / name}.summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
