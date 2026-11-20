"""The real Groq client against a mocked HTTP layer: no network, no key, no quota."""

import json

import httpx
import pytest

from app.llm import GroqClient, LLMRateLimited, LLMTimeout, LLMUnavailable

MESSAGES = [{"role": "user", "content": "hi"}]


def client_with(handler, reasoning_effort="none") -> GroqClient:
    return GroqClient("test-key", "some-model", "https://groq.test/v1", reasoning_effort,
                      transport=httpx.MockTransport(handler))


def ok(content: str):
    return lambda request: httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_returns_message_content_and_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return ok('{"status": "not_covered"}')(request)

    assert client_with(handler).complete(MESSAGES, 2.0) == '{"status": "not_covered"}'
    assert seen["url"] == "https://groq.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "some-model" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "none"


def test_reasoning_effort_is_omitted_when_empty():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return ok("{}")(request)

    client_with(handler, reasoning_effort="").complete(MESSAGES, 2.0)
    assert "reasoning_effort" not in seen["body"]


@pytest.mark.parametrize("status, error", [(429, LLMRateLimited), (500, LLMUnavailable),
                                           (503, LLMUnavailable), (401, LLMUnavailable)])
def test_http_errors_map_to_llm_errors(status, error):
    with pytest.raises(error):
        client_with(lambda r: httpx.Response(status, json={"error": "x"})).complete(MESSAGES, 2.0)


def test_timeout_maps_to_llm_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeout):
        client_with(handler).complete(MESSAGES, 0.1)


def test_network_error_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(LLMUnavailable):
        client_with(handler).complete(MESSAGES, 2.0)


@pytest.mark.parametrize("payload", [{"choices": []}, {"unexpected": True}, "not json at all"])
def test_unexpected_response_shape_maps_to_unavailable(payload):
    def handler(request):
        if isinstance(payload, str):
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json=payload)

    with pytest.raises(LLMUnavailable):
        client_with(handler).complete(MESSAGES, 2.0)


def test_missing_key_is_refused():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient("", "m", "https://groq.test/v1")
