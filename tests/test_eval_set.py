"""The evaluation set and its scoring are code too: test them.

A typo in an expected section id would silently turn a correct answer into a
"failure" (or worse, a wrong one into a pass), so the answer key is checked
against the actual policy documents.
"""

import pytest

from app.ingestion import parse_sections
from evaluation.run_eval import DEFAULT_CASES, load_cases
from evaluation.scoring import missing_facts, score_case
from tests.conftest import read_policy

CASES = load_cases(DEFAULT_CASES)
REQUIRED_CATEGORIES = {"direct", "paraphrase", "not_covered", "cross_company", "injection", "version"}


def policy_sections(company: str, stage: str) -> set[str]:
    version = "v2" if company == "nimbus" and stage == "after_nimbus_v2" else "v1"
    return {s.section_id for s in parse_sections(read_policy(f"{company}/transport-policy.{version}.md"))}


def test_eval_set_meets_the_brief():
    assert len(CASES) >= 10
    assert REQUIRED_CATEGORIES <= {c["expected"]["category"] for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_sections_exist_in_that_companys_policy(case):
    sections = set(case["expected"].get("sections", []))
    available = policy_sections(case["input"]["company"], case["input"].get("stage", "initial"))
    assert sections <= available, f"{sorted(sections - available)} not in {case['input']['company']}'s policy"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_is_well_formed(case):
    exp = case["expected"]
    statuses = exp["status"] if isinstance(exp["status"], list) else [exp["status"]]
    assert set(statuses) <= {"answered", "not_covered"}
    if statuses == ["answered"]:
        assert exp.get("sections"), "an answerable case must say which sections support it"
        assert exp.get("facts"), "an answerable case must say what the answer has to contain"
    if statuses == ["not_covered"]:
        assert not exp.get("facts")


# --- scoring rules -------------------------------------------------------------------

BASE_ACTUAL = {
    "status": "answered", "answer": "An escort is mandatory after 8:00 PM for women employees.",
    "cited_sections": ["4.1"], "cited_versions": [1], "sent_sections": ["4.1", "4.2"], "sent_versions": [1],
    "candidate_sections": ["4.1"], "foreign_chunks": 0, "llm_calls": 1,
}
EXPECTED = {"category": "direct", "status": "answered", "sections": ["4.1", "4.2"],
            "facts": [["8:00 pm", "8 pm"], ["woman", "women"]], "must_not_contain": ["10:00 pm"]}


def score(mode="llm", expected=None, **changes):
    return score_case(expected or EXPECTED, {**BASE_ACTUAL, **changes}, mode)


def test_correct_answer_passes():
    passed, checks, notes = score()
    assert passed and notes == []


@pytest.mark.parametrize("changes, failed_check", [
    ({"status": "not_covered", "answer": "Please contact your transport desk."}, "status"),
    ({"cited_sections": ["5.1"]}, "citation"),
    ({"cited_sections": []}, "citation"),
    ({"answer": "An escort is mandatory after 8 PM."}, "facts"),  # lost "women"
    ({"answer": "Escort for women after 8 PM, or 10:00 PM at other firms."}, "no_forbidden_text"),
    ({"foreign_chunks": 1}, "isolation"),
    ({"llm_calls": 3}, "llm_calls<=2"),
])
def test_each_rule_can_fail_a_case(changes, failed_check):
    passed, checks, _ = score(**changes)
    assert not passed and checks[failed_check] is False


def test_fact_alternatives_and_typography():
    assert missing_facts("Escort after 8 PM for women", [["8:00 pm", "8 pm"], ["women"]]) == []
    assert missing_facts("Escort after 9 PM", [["8:00 pm", "8 pm"]]) == ["8:00 pm | 8 pm"]


def test_version_must_match_when_expected():
    exp = {**EXPECTED, "version": 2}
    assert score(expected=exp, cited_versions=[2])[0]
    assert not score(expected=exp, cited_versions=[1])[0]


def test_fake_mode_scores_retrieval_not_answer_text():
    passed, checks, notes = score(mode="fake", answer="first sentence of the top chunk, unrelated")
    assert passed and "facts" not in checks and checks["retrieval_hit@3"]
    assert "answer not scored (fake LLM)" in notes
    passed, checks, _ = score(mode="fake", sent_sections=["5.1"])
    assert not passed and checks["retrieval_hit@3"] is False


def test_fake_mode_scores_statuses_decided_without_the_llm():
    refusal = {"category": "cross_company", "status": "not_covered"}
    assert score(mode="fake", expected=refusal, status="not_covered", answer="contact your transport desk",
                 llm_calls=0)[0]
    assert not score(mode="fake", expected=refusal, status="answered", llm_calls=1)[0]
