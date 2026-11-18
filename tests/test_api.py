import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.generation.llm import LLMError

client = TestClient(api.app)     # no `with`: skips the model warm-up in tests


def fake_result(found=True):
    return {
        "question": "q", "answer": "Use rollout undo [1].", "found": found,
        "sources": [{"n": 1, "section": "Deployments > Rolling Back", "url": "https://k/d/"}],
        "retrieved": [{"section": "Deployments > Rolling Back", "url": "https://k/d/", "text": "t" * 3000,
                       "dense_rank": 1, "bm25_rank": None}],
        "llm": {"model": "m", "prompt_tokens": 100, "completion_tokens": 20, "seconds": 0.5},
        "retrieval_seconds": 0.2,
    }


def test_health_ok(monkeypatch):
    monkeypatch.setattr(api, "_probe", lambda: {"vectors": 10, "chunks": 10})
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["vectors"] == 10


def test_health_unavailable(monkeypatch):
    def boom():
        raise ConnectionError("qdrant down")
    monkeypatch.setattr(api, "_probe", boom)
    assert client.get("/health").status_code == 503


def test_ask_success_shape(monkeypatch):
    monkeypatch.setattr(api, "rag_answer", lambda q, k: fake_result())
    body = client.post("/ask", json={"question": "  how to roll back?  "}).json()
    assert body["found"] and body["sources"][0]["n"] == 1
    assert body["retrieved"][0]["n"] == 1 and len(body["retrieved"][0]["text"]) == 1500   # truncated
    assert body["rewritten_question"] is None and set(body["timings"]) == {"rewrite_s", "retrieval_s", "generation_s", "total_s"}


def test_ask_passes_question_and_k(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "rag_answer", lambda q, k: seen.update(q=q, k=k) or fake_result())
    client.post("/ask", json={"question": "  hello  ", "k": 3})
    assert seen == {"q": "hello", "k": 3}


def test_follow_up_uses_rewritten_question(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "contextualize", lambda q, h: "How do I check Deployment rollout history?")
    monkeypatch.setattr(api, "rag_answer", lambda q, k: seen.update(q=q) or fake_result())
    body = client.post("/ask", json={"question": "and its history?", "history": [
        {"role": "user", "content": "roll back a deployment"}, {"role": "assistant", "content": "ok"}]}).json()
    assert seen["q"] == "How do I check Deployment rollout history?"
    assert body["rewritten_question"] == "How do I check Deployment rollout history?"


def test_rewrite_failure_falls_back_to_original(monkeypatch):
    def boom(q, h):
        raise LLMError("rate limited")
    seen = {}
    monkeypatch.setattr(api, "contextualize", boom)
    monkeypatch.setattr(api, "rag_answer", lambda q, k: seen.update(q=q) or fake_result())
    r = client.post("/ask", json={"question": "and its history?", "history": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and seen["q"] == "and its history?" and r.json()["rewritten_question"] is None


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": "ok", "k": 0},
    {"question": "ok", "k": 11}, {"question": "ok", "history": [{"role": "system", "content": "x"}]}, {},
])
def test_validation_errors(payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_error_maps_to_502(monkeypatch):
    def boom(q, k):
        raise LLMError("HTTP 401: bad key")
    monkeypatch.setattr(api, "rag_answer", boom)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 502 and "401" in r.json()["detail"]


def test_retrieval_error_maps_to_503_without_leaking_details(monkeypatch):
    def boom(q, k):
        raise ConnectionError("secret-host:6333 refused")
    monkeypatch.setattr(api, "rag_answer", boom)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 503 and "secret-host" not in r.text
