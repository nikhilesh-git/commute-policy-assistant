"""Deterministic scoring for one eval case. No LLM-as-judge: same input, same score.

A case's `expected` block:
  category          direct | paraphrase | not_covered | cross_company | injection | version
  status            "answered" | "not_covered", or a list of acceptable statuses
  sections          section ids the answer may cite (and retrieval should find)
  facts             every item must appear in the answer; an item may be a list of
                    alternatives, e.g. [["8:00 pm", "8 pm"], ["escort", "guard"]]
  must_not_contain  strings that must never appear (usually another company's value)
  version           policy version every citation must come from

Two modes:
  llm   (real model)  the answer itself is scored: status, citations, facts, versions
  fake  (no key)      the fake LLM cannot understand questions, so only what is
                      independent of the model is scored: retrieval found the right
                      section, refusals, version filtering, isolation, call limit.
"""

from app.guards import normalise

FALLBACK_HINT = "transport desk"


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def section_of(citation_section: str) -> str:
    """'4.1 Escort for women employees' -> '4.1'"""
    return citation_section.split(" ", 1)[0]


def missing_facts(answer: str, facts: list) -> list[str]:
    text = normalise(answer)
    return [" | ".join(_as_list(f)) for f in facts
            if not any(normalise(alt) in text for alt in _as_list(f))]


def forbidden_found(answer: str, forbidden: list[str]) -> list[str]:
    text = normalise(answer)
    return [s for s in forbidden if normalise(s) in text]


def score_case(expected: dict, actual: dict, mode: str) -> tuple[bool, dict[str, bool], list[str]]:
    """Return (passed, checks, notes)."""
    checks: dict[str, bool] = {}
    notes: list[str] = []
    allowed = _as_list(expected["status"])
    sections = set(expected.get("sections", []))
    is_refusal = expected["category"] == "cross_company" and allowed == ["not_covered"]

    # --- always: things no model can excuse ------------------------------------
    checks["isolation"] = actual["foreign_chunks"] == 0
    checks["llm_calls<=2"] = actual["llm_calls"] <= 2
    bad = forbidden_found(actual["answer"], expected.get("must_not_contain", []))
    checks["no_forbidden_text"] = not bad
    if bad:
        notes.append(f"forbidden text in answer: {bad}")

    if mode == "fake":
        # A status reached with 0 LLM calls was decided by code (refusal, relevance
        # gate), so it is scorable even with a fake model. Refusals must always be.
        if is_refusal or actual["llm_calls"] == 0:
            checks["status"] = actual["status"] in allowed
            if not checks["status"]:
                notes.append(f"status {actual['status']!r}, expected {allowed}")
        if sections and "answered" in allowed:
            checks["retrieval_hit@3"] = bool(sections & set(actual["sent_sections"]))
            if not checks["retrieval_hit@3"]:
                notes.append(f"expected section {sorted(sections)} not in top 3 {actual['sent_sections']}")
        if "version" in expected and actual["sent_versions"]:
            checks["version"] = set(actual["sent_versions"]) == {expected["version"]}
        if actual["llm_calls"] > 0:
            notes.append("answer not scored (fake LLM)")
    else:
        checks["status"] = actual["status"] in allowed
        if not checks["status"]:
            notes.append(f"status {actual['status']!r}, expected {allowed}")
        if actual["status"] == "answered":
            cited = set(actual["cited_sections"])
            if sections:
                checks["citation"] = bool(cited) and cited <= sections
                if not checks["citation"]:
                    notes.append(f"cited {sorted(cited)}, expected within {sorted(sections)}")
            missing = missing_facts(actual["answer"], expected.get("facts", []))
            checks["facts"] = not missing
            if missing:
                notes.append(f"missing facts: {missing}")
            if "version" in expected:
                checks["version"] = set(actual["cited_versions"]) == {expected["version"]}
                if not checks["version"]:
                    notes.append(f"cited versions {actual['cited_versions']}, expected {expected['version']}")
        elif actual["status"] == "not_covered":
            checks["points_to_desk"] = FALLBACK_HINT in actual["answer"].lower()

    passed = all(checks.values())
    return passed, checks, notes
