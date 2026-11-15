"""Groq client: caching, retries, clear errors. A fake SDK client is used (no network)."""
import httpx
import pytest
from groq import AuthenticationError, NotFoundError, RateLimitError

from finrisk.genai.groq_client import GroqLLM, LLMError


def _resp(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "http://x"))


class _Msg:
    def __init__(self, text): self.message = type("M", (), {"content": text})()


class FakeSDK:
    """Mimics client.chat.completions.create(...); `script` is a list of results or exceptions."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        outer = self
        class _Completions:
            def create(self, **kw):
                outer.calls += 1
                item = outer.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return type("R", (), {"choices": [_Msg(item)]})()
        self.chat = type("C", (), {"completions": _Completions()})()


def _llm(script, tmp_path, sleeps):
    return GroqLLM(model="test-model", cache_dir=tmp_path, client=FakeSDK(script), max_retries=4,
                   min_interval=0, sleep=sleeps.append)


MSG = [{"role": "user", "content": "hi"}]


def test_second_identical_call_is_served_from_the_cache(tmp_path):
    sleeps = []
    llm = _llm(["hello"], tmp_path, sleeps)
    assert llm.chat(MSG) == "hello"
    assert llm.chat(MSG) == "hello"
    assert llm.client.calls == 1 and llm.cache_hits == 1


def test_different_model_or_messages_do_not_share_cache_entries(tmp_path):
    llm = _llm(["a", "b"], tmp_path, [])
    assert llm.chat(MSG) == "a"
    assert llm.chat([{"role": "user", "content": "other"}]) == "b"


def test_rate_limit_is_retried_with_backoff_then_succeeds(tmp_path):
    sleeps = []
    err = RateLimitError("slow down", response=_resp(429), body=None)
    llm = _llm([err, err, "finally"], tmp_path, sleeps)
    assert llm.chat(MSG) == "finally"
    assert llm.client.calls == 3
    assert sleeps == [2.0, 4.0]                      # exponential backoff


def test_retry_after_header_is_honoured(tmp_path):
    sleeps = []
    err = RateLimitError("slow down", response=_resp(429, {"retry-after": "7"}), body=None)
    llm = _llm([err, "ok"], tmp_path, sleeps)
    llm.chat(MSG)
    assert sleeps == [7.0]


def test_gives_up_after_max_retries_with_a_helpful_message(tmp_path):
    err = RateLimitError("slow down", response=_resp(429), body=None)
    llm = _llm([err] * 4, tmp_path, [])
    with pytest.raises(LLMError, match="rate limit"):
        llm.chat(MSG)


def test_bad_key_gives_a_clear_error_and_no_retry(tmp_path):
    err = AuthenticationError("nope", response=_resp(401), body=None)
    llm = _llm([err, "never"], tmp_path, [])
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        llm.chat(MSG)
    assert llm.client.calls == 1


def test_retired_model_points_to_list_models(tmp_path):
    err = NotFoundError("model gone", response=_resp(404), body=None)
    llm = _llm([err], tmp_path, [])
    with pytest.raises(LLMError, match="--list-models"):
        llm.chat(MSG)


def test_missing_key_is_reported_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr("finrisk.genai.groq_client.get_settings",
                        lambda: type("S", (), {"groq_api_key": "your_key_here", "groq_model": "m"})())
    with pytest.raises(LLMError, match="No Groq API key"):
        GroqLLM(cache_dir=tmp_path)
