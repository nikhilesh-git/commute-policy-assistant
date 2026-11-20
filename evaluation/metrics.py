"""Aggregate metrics over scored eval rows. Every number here is defined in EVALUATION.md."""

from app.metrics import percentile


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def retrieval_metrics(rows: list[dict]) -> dict:
    """Over cases that expect an answer from specific sections."""
    hit1 = hit3 = 0
    rr = 0.0
    n = 0
    for r in rows:
        exp = r["expected"]
        statuses = exp["status"] if isinstance(exp["status"], list) else [exp["status"]]
        if not exp.get("sections") or "answered" not in statuses:
            continue
        n += 1
        wanted = set(exp["sections"])
        cands = r["actual"]["candidate_sections"]
        hit1 += bool(cands) and cands[0] in wanted
        hit3 += bool(wanted & set(r["actual"]["sent_sections"]))
        rank = next((i for i, s in enumerate(cands, 1) if s in wanted), None)
        rr += 1 / rank if rank else 0.0
    return {"cases": n, "hit@1": _rate(hit1, n), "hit@3_sent_to_llm": _rate(hit3, n),
            "mrr@8": round(rr / n, 3) if n else None}


def answer_metrics(rows: list[dict]) -> dict:
    """Only meaningful with a real LLM."""
    def statuses(r):
        s = r["expected"]["status"]
        return s if isinstance(s, list) else [s]

    status_ok = sum(r["actual"]["status"] in statuses(r) for r in rows)
    should_answer = [r for r in rows if statuses(r) == ["answered"]]
    should_refuse = [r for r in rows if statuses(r) == ["not_covered"]]
    said_nc = [r for r in rows if r["actual"]["status"] == "not_covered"]
    answered = [r for r in rows if r["actual"]["status"] == "answered" and r["expected"].get("sections")]
    cite_ok = sum(set(r["actual"]["cited_sections"]) <= set(r["expected"]["sections"])
                  and bool(r["actual"]["cited_sections"]) for r in answered)
    return {
        "status_accuracy": _rate(status_ok, len(rows)),
        "answerable_fully_correct": _rate(sum(r["pass"] for r in should_answer), len(should_answer)),
        "not_covered_recall": _rate(sum(r["actual"]["status"] == "not_covered" for r in should_refuse),
                                    len(should_refuse)),
        "not_covered_precision": _rate(sum(statuses(r) != ["answered"] for r in said_nc), len(said_nc)),
        "citation_accuracy": _rate(cite_ok, len(answered)),
        "try_again": sum(r["actual"]["status"] == "try_again" for r in rows),
    }


def safety_metrics(rows: list[dict], mode: str) -> dict:
    cross = [r for r in rows if r["expected"]["category"] == "cross_company"]
    inj = [r for r in rows if r["expected"]["category"] == "injection"]
    return {
        "foreign_chunks_retrieved": sum(r["actual"]["foreign_chunks"] for r in rows),
        "cross_company_failures": sum(not r["pass"] for r in cross),
        # The fake LLM only echoes text, so "did it obey the injection" means nothing there.
        "injection_obeyed": (sum(not r["checks"].get("no_forbidden_text", True) for r in inj)
                             if mode == "llm" else None),
        "max_llm_calls": max((r["actual"]["llm_calls"] for r in rows), default=0),
    }


def latency_metrics(all_requests: list[dict]) -> dict:
    """Over every request in every repeat (server-side total, LLM call included)."""
    total = [r["latency_ms"] for r in all_requests]
    answered = [r["latency_ms"] for r in all_requests if r["status"] == "answered"]
    with_llm = [r["latency_ms"] for r in all_requests if r["llm_calls"] > 0]
    calls = [r["llm_calls"] for r in all_requests]
    return {
        "requests": len(total),
        "p50_ms": round(percentile(total, 50)) if total else None,
        "p95_ms": round(percentile(total, 95)) if total else None,
        "max_ms": round(max(total)) if total else None,
        "p95_answered_ms": round(percentile(answered, 95)) if answered else None,
        "p95_with_llm_call_ms": round(percentile(with_llm, 95)) if with_llm else None,
        "mean_llm_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "max_llm_calls": max(calls, default=0),
        "under_3s": _rate(sum(t < 3000 for t in total), len(total)),
    }


def by_category(rows: list[dict]) -> dict:
    out: dict[str, str] = {}
    for cat in sorted({r["expected"]["category"] for r in rows}):
        sub = [r for r in rows if r["expected"]["category"] == cat]
        out[cat] = f"{sum(r['pass'] for r in sub)}/{len(sub)}"
    return out


def summarise(rows: list[dict], all_requests: list[dict], mode: str, unstable: list[str]) -> dict:
    summary = {
        "passed": f"{sum(r['pass'] for r in rows)}/{len(rows)}",
        "by_category": by_category(rows),
        "retrieval": retrieval_metrics(rows),
        "safety": safety_metrics(rows, mode),
        "latency": latency_metrics(all_requests),
        "cases_with_different_result_across_repeats": unstable,
    }
    if mode == "llm":
        summary["answers"] = answer_metrics(rows)
    return summary
