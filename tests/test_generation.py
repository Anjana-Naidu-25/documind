import pytest
import requests

from src.generation import llm as llm_mod
from src.generation.prompt import NOT_FOUND, build_messages
from src.generation.rag import answer, extract_citations, is_not_found

CHUNKS = [
    {"text": "Pods > Lifecycle\n\nA pod restarts containers per restartPolicy.", "section": "Pods > Lifecycle", "url": "https://k/pods/"},
    {"text": "Jobs > Cleanup\n\nJobs clean up finished pods.", "section": "Jobs > Cleanup", "url": "https://k/jobs/"},
]


def test_prompt_numbers_passages_and_contains_rules():
    msgs = build_messages("why?", CHUNKS)
    assert msgs[0]["role"] == "system" and NOT_FOUND in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] https://k/pods/" in user and "[2] https://k/jobs/" in user and user.endswith("Question: why?")


def test_extract_citations_ignores_invalid_numbers():
    assert extract_citations("Yes [1]. Also [2][2] and [7] and [0].", 2) == [1, 2]
    assert extract_citations("no citations", 2) == []


def test_not_found_detection_tolerates_apostrophe_and_period():
    assert is_not_found("I couldn’t find this in the documentation")
    assert not is_not_found("Pods restart containers [1].")


def _fake_llm(text):
    return lambda messages: {"text": text, "model": "m", "prompt_tokens": 10, "completion_tokens": 5, "seconds": 0.1}


def test_answer_attaches_only_cited_sources():
    res = answer("q", retriever=lambda q, k: CHUNKS, llm=_fake_llm("Containers restart per policy [1]."))
    assert res["found"] and [s["n"] for s in res["sources"]] == [1]
    assert res["sources"][0]["url"] == "https://k/pods/"


def test_answer_not_found_has_no_sources():
    res = answer("q", retriever=lambda q, k: CHUNKS, llm=_fake_llm(NOT_FOUND))
    assert not res["found"] and res["sources"] == []


class FakeResp:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, text

    def json(self):
        return self._body


OK = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def test_chat_retries_on_429_then_succeeds(monkeypatch):
    calls, sleeps = [], []
    responses = iter([FakeResp(429, headers={"retry-after": "2"}), FakeResp(200, OK)])
    monkeypatch.setattr(llm_mod, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: calls.append(kw) or next(responses))
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    out = llm_mod.chat([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert out["text"] == "hello" and len(calls) == 2 and sleeps == [2.0]
    assert calls[0]["json"]["reasoning_effort"]           # sent for gpt-oss models


def test_chat_raises_on_client_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: FakeResp(401, text="bad key"))
    with pytest.raises(llm_mod.LLMError, match="401"):
        llm_mod.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_empty_answer(monkeypatch):
    empty = {"choices": [{"message": {"content": None}}]}
    monkeypatch.setattr(llm_mod, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **kw: FakeResp(200, empty))
    with pytest.raises(llm_mod.LLMError, match="Empty"):
        llm_mod.chat([{"role": "user", "content": "hi"}])


def test_chat_requires_key(monkeypatch):
    monkeypatch.setattr(llm_mod, "LLM_API_KEY", "")
    with pytest.raises(llm_mod.LLMError, match="API key"):
        llm_mod.chat([{"role": "user", "content": "hi"}])
