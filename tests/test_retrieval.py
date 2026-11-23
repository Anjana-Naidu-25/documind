from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.hybrid import hybrid_search, rrf


def test_tokenizer_keeps_identifier_and_parts():
    t = tokenize("terminationGracePeriodSeconds")
    assert "terminationgraceperiodseconds" in t
    assert {"termination", "grace", "period", "seconds"} <= set(t)


def test_tokenizer_handles_flags_and_stopwords():
    t = tokenize("kubectl rollout undo --to-revision")
    assert "kubectl" in t and "undo" in t and "revision" in t
    assert "the" not in tokenize("the pod")


def test_bm25_finds_exact_identifier():
    recs = [
        {"id": "a", "text": "A pod has a terminationGracePeriodSeconds field for shutdown."},
        {"id": "b", "text": "Jobs use activeDeadlineSeconds to limit run time."},
        {"id": "c", "text": "Services expose pods to the network."},
    ]
    top, score = BM25Index(recs).search("terminationGracePeriodSeconds", k=3)[0]
    assert top["id"] == "a" and score > 0


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["c", "a", "d"]]))
    assert fused["a"] > fused["b"] and fused["a"] > fused["d"]   # a is high in both lists
    assert rrf([["a"], ["b"]])[0][1] == rrf([["a"], ["b"]])[1][1]   # equal ranks -> equal score


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_hybrid_merges_with_stubbed_retrievers(monkeypatch):
    import src.retrieval.bm25 as bm
    recs = [{"id": i, "text": f"chunk {i}", "section": f"S{i}", "url": f"u{i}"} for i in "abcd"]
    monkeypatch.setattr(bm, "_index", BM25Index(recs))
    res = hybrid_search("q", k=3, dense_fn=lambda q, n: ["a", "b", "c"], bm25_fn=lambda q, n: ["c", "a", "d"])
    assert [r["id"] for r in res] == ["a", "c", "b"]
    assert res[0]["dense_rank"] == 1 and res[0]["bm25_rank"] == 2
    assert res[2]["id"] == "b" and res[2]["bm25_rank"] is None   # b was only found by the dense retriever


def test_hybrid_rerank_reorders_and_keeps_rank_info(monkeypatch):
    import src.retrieval.bm25 as bm
    recs = [{"id": i, "text": f"chunk {i}", "section": f"S{i}", "url": f"u{i}"} for i in "abcd"]
    monkeypatch.setattr(bm, "_index", BM25Index(recs))
    # fused order is a, c, b, d; the stub reranker prefers d, then b
    def stub(query, cands, top_k):
        order = {"d": 0, "b": 1}
        return sorted(cands, key=lambda r: order.get(r["id"], 9))[:top_k]
    res = hybrid_search("q", k=2, n_candidates=4, rerank=True, rerank_fn=stub,
                        dense_fn=lambda q, n: ["a", "b", "c"], bm25_fn=lambda q, n: ["c", "a", "d"])
    assert [r["id"] for r in res] == ["d", "b"]
    assert res[0]["bm25_rank"] == 3 and res[0]["dense_rank"] is None and res[0]["score"] > 0
