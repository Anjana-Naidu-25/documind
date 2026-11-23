import json

from src.evaluation import generation_eval as ge
from src.evaluation import make_testset as mt
from src.evaluation.dataset import COMPARISON, OUT_OF_SCOPE, curated_rows
from src.evaluation.metrics import coverage_at_k, dedupe_keep_order, hit_at_k, reciprocal_rank
from src.evaluation.retrieval_eval import evaluate, format_report
from src.retrieval.rerank import rerank


# ---------- metrics ----------
def test_hit_and_rr():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, 3) == 1.0 and hit_at_k(ranked, {"c"}, 2) == 0.0
    assert reciprocal_rank(ranked, {"b", "d"}) == 0.5 and reciprocal_rank(ranked, {"zzz"}) == 0.0
    assert reciprocal_rank(ranked, {"d"}, k=3) == 0.0          # outside the cut-off


def test_coverage():
    assert coverage_at_k(["u1", "u3"], {"u1", "u2"}, 5) == 0.5
    assert coverage_at_k([], set(), 5) == 0.0


def test_dedupe_keeps_first_occurrence():
    assert dedupe_keep_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


# ---------- curated data ----------
def test_curated_rows_are_well_formed():
    rows = curated_rows()
    assert len(rows) == len(COMPARISON) + len(OUT_OF_SCOPE)
    cmp = [r for r in rows if r["type"] == "comparison"]
    assert all(len(r["expected_urls"]) == 2 and r["expected_urls"][0].startswith("https://kubernetes.io/docs/") for r in cmp)
    assert all(r["should_refuse"] for r in rows if r["type"] == "out_of_scope")
    assert len({r["id"] for r in rows}) == len(rows)


# ---------- test set generation ----------
def test_parse_json_tolerates_prose_and_fences():
    assert mt.parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert mt.parse_json("no json here") is None and mt.parse_json("{broken") is None


def _rec(i, url, text, section="Page > Topic"):
    return {"id": f"id{i}", "url": url, "section": section, "text": text, "tokens": 150}


def test_selection_uses_each_page_once_and_classifies():
    recs = [_rec(i, f"https://k/p{i}/", "Explains a concept in words only. " * 10) for i in range(6)]
    recs += [_rec(10 + i, f"https://k/h{i}/", "Steps:\n```shell\nkubectl get pods\n```\n" * 3) for i in range(4)]
    recs += [_rec(20, "https://k/p0/", "Another chunk of page p0 " * 10)]         # same page as a concept chunk
    recs += [_rec(30, "https://k/skip/", "x " * 100, section="Page > Before you begin")]
    picked = mt.select_chunks(recs, per_type=3, seed=1)
    urls = [c["url"] for _, c in picked]
    assert len(urls) == len(set(urls)) and "https://k/skip/" not in urls
    assert {t for t, _ in picked} >= {"howto", "concept"}


def test_build_with_fake_llm(monkeypatch):
    monkeypatch.setattr(mt.time, "sleep", lambda s: None)
    recs = [_rec(i, f"https://k/p{i}/", "Plain explanation text. " * 20) for i in range(4)]
    replies = iter(['{"question": "What does this Kubernetes thing do?", "reference_answer": "It does X."}', "garbage",
                    '{"question": "short", "reference_answer": "x"}', '{"question": "How is the Kubernetes widget configured?", "reference_answer": "Y."}'])
    rows = mt.build(recs, per_type=4, seed=0, llm=lambda m: {"text": next(replies)})
    assert len(rows) == 2 and rows[0]["expected_chunk_ids"] and rows[0]["expected_urls"][0].startswith("https://k/")


# ---------- retrieval evaluation ----------
TESTSET = [
    {"id": "a", "type": "concept", "question": "q1", "expected_urls": ["u1"], "expected_chunk_ids": ["c1"], "should_refuse": False},
    {"id": "b", "type": "howto", "question": "q2", "expected_urls": ["u2"], "expected_chunk_ids": ["c2"], "should_refuse": False},
    {"id": "c", "type": "comparison", "question": "q3", "expected_urls": ["u3", "u4"], "expected_chunk_ids": [], "should_refuse": False},
    {"id": "d", "type": "out_of_scope", "question": "q4", "expected_urls": [], "expected_chunk_ids": [], "should_refuse": True},
]
ID2URL = {"c1": "u1", "c1b": "u1", "c2": "u2", "x": "ux", "c3": "u3", "c4": "u4"}


def test_evaluate_computes_levels_and_skips_out_of_scope():
    rankers = {
        "good": lambda q: {"q1": ["c1", "x"], "q2": ["x", "c2"], "q3": ["c3", "c4"]}[q],
        "bad": lambda q: {"q1": ["x", "c1b"], "q2": ["x"], "q3": ["x", "c3"]}[q],
    }
    out = evaluate(TESTSET, rankers, ID2URL)["summary"]
    assert out["good"]["overall"]["n"] == 2 and out["good"]["overall"]["chunk_hit@1"] == 0.5
    assert out["good"]["overall"]["chunk_mrr"] == 0.75 and out["good"]["comparison"]["all@5"] == 1.0
    assert out["bad"]["overall"]["chunk_hit@5"] == 0.0          # c1b is the wrong chunk (strict)...
    assert out["bad"]["overall"]["page_hit@5"] == 0.5           # ...but the right page (lenient)
    assert out["bad"]["comparison"]["cov@5"] == 0.5
    assert "| good |" in format_report(out) and "Comparison questions" in format_report(out)


# ---------- reranker ----------
def test_rerank_orders_by_score_and_trims():
    recs = [{"id": "a", "text": "aaa"}, {"id": "b", "text": "bbb"}, {"id": "c", "text": "ccc"}]
    out = rerank("q", recs, top_k=2, predict_fn=lambda pairs: [0.1, 0.9, 0.5])
    assert [r["id"] for r in out] == ["b", "c"] and out[0]["rerank_score"] == 0.9
    assert rerank("q", [], predict_fn=lambda p: []) == []


# ---------- generation evaluation ----------
def fake_answer(found=True, text="Use X [1]."):
    return {"answer": text, "found": found, "sources": [{"n": 1, "section": "S", "url": "u1"}],
            "retrieved": [{"url": "u1", "text": "passage text"}], "retrieval_seconds": 0.1,
            "llm": {"seconds": 0.5, "prompt_tokens": 100, "completion_tokens": 20}}


JUDGE_OK = lambda m: {"text": '{"faithful": true, "unsupported_claims": [], "verdict": "correct", "reason": "ok"}', "prompt_tokens": 50, "completion_tokens": 10}
ROW = {"id": "a", "type": "concept", "question": "q", "reference_answer": "ref", "should_refuse": False}


def test_run_one_judged_answer():
    rec = ge.run_one(ROW, lambda q: fake_answer(), JUDGE_OK)
    assert rec["faithful"] is True and rec["verdict"] == "correct" and rec["false_refusal"] is False and rec["judge_tokens"] == 60


def test_run_one_false_refusal_skips_judge():
    def never(m):
        raise AssertionError("judge must not be called")
    rec = ge.run_one(ROW, lambda q: fake_answer(found=False, text="I couldn't find this in the documentation."), never)
    assert rec["false_refusal"] is True and rec["verdict"] == "incorrect"


def test_run_one_out_of_scope():
    oos = {"id": "o", "type": "out_of_scope", "question": "capital of France?", "reference_answer": "", "should_refuse": True}
    assert ge.run_one(oos, lambda q: fake_answer(found=False), JUDGE_OK)["correct_refusal"] is True
    assert ge.run_one(oos, lambda q: fake_answer(found=True), JUDGE_OK)["correct_refusal"] is False


def test_judge_parse_failure_is_flagged():
    j = ge.judge(ROW, fake_answer(), lambda m: {"text": "I think it is fine."})
    assert j["judge_parse_failed"] is True and j["verdict"] is None


def test_sample_covers_types_round_robin():
    ts = [{"id": f"{t}{i}", "type": t, "should_refuse": t == "out_of_scope"} for t in ge.TYPE_ORDER for i in range(5)]
    s = ge.sample(ts, 10)
    assert len(s) == 10 and {r["type"] for r in s} == set(ge.TYPE_ORDER)
    assert len(ge.sample(ts, 1000)) == len(ts)


def test_summarize_numbers():
    judged = lambda i, faithful, verdict: {"id": i, "type": "concept", "question": f"q{i}", "answer": "a", "found": True,
                                         "false_refusal": False, "faithful": faithful, "verdict": verdict,
                                         "unsupported_claims": ["claim"], "seconds": 1.0, "tokens": 100}
    rows = [judged(1, True, "correct"), judged(2, False, "partial"),
            {"id": "r", "type": "concept", "question": "q", "answer": "x", "found": False, "false_refusal": True,
             "faithful": None, "verdict": "incorrect", "unsupported_claims": [], "seconds": 1.0, "tokens": 100},
            {"id": "o", "type": "out_of_scope", "question": "q", "answer": "x", "found": False, "correct_refusal": True,
             "seconds": 1.0, "tokens": 100}]
    text = ge.summarize(rows)
    assert "Correct (judge)        : 33%" in text and "Faithful (judge)       : 50%" in text
    assert "False refusals         : 33%" in text and "Correct refusals       : 100%" in text
    assert "Unfaithful answers" in text
