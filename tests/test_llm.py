import pytest

from app.llm import FakeLLM, LLMBudget, LLMBudgetExceeded, LLMTimeout


def test_budget_allows_at_most_max_calls():
    budget = LLMBudget(FakeLLM(), max_calls=2)
    budget.complete([], 1.0)
    budget.complete([], 1.0)
    with pytest.raises(LLMBudgetExceeded):
        budget.complete([], 1.0)
    assert budget.calls == 2


def test_fake_llm_follows_script_and_records_prompts():
    fake = FakeLLM(script=[LLMTimeout("slow"), '{"status": "not_covered"}'])
    with pytest.raises(LLMTimeout):
        fake.complete([{"role": "user", "content": "q1"}], 1.0)
    assert fake.complete([{"role": "user", "content": "q2"}], 1.0) == '{"status": "not_covered"}'
    assert [m[0]["content"] for m in fake.received] == ["q1", "q2"]
