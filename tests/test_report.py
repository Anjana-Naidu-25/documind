import json

import pytest

from src.evaluation import generation_eval as ge
from src.evaluation import make_report as mr


def row(i, verdict="correct", faithful=True, false_refusal=False):
    return {"id": str(i), "type": "concept", "question": "q", "answer": "a", "found": not false_refusal,
            "false_refusal": false_refusal, "faithful": None if false_refusal else faithful, "verdict": verdict,
            "unsupported_claims": [], "seconds": 2.0, "tokens": 1000}


ROWS = [row(1), row(2, "partial", False), row(3, "incorrect", false_refusal=True), row(4),
        {"id": "o1", "type": "out_of_scope", "question": "q", "answer": "x", "found": False, "correct_refusal": True, "seconds": 1.0, "tokens": 500},
        {"id": "o2", "type": "out_of_scope", "question": "q", "answer": "x", "found": True, "correct_refusal": False, "seconds": 1.0, "tokens": 500}]


def test_stats_counts():
    s = ge.stats(ROWS)
    assert (s["n_answerable"], s["n_judged"], s["n_oos"]) == (4, 3, 2)
    assert (s["n_correct"], s["n_partial"], s["n_faithful"], s["n_false_refusal"], s["n_correct_refusal"]) == (2, 1, 2, 1, 1)


def test_stats_agree_with_printed_summary():
    s, text = ge.stats(ROWS), ge.summarize(ROWS)
    assert f"{s['n_correct'] / s['n_answerable']:.0%}" in text.split("Correct (judge)")[1].splitlines()[0]
    assert f"{s['n_faithful'] / s['n_judged']:.0%}" in text.split("Faithful (judge)")[1].splitlines()[0]
    assert f"{s['n_correct_refusal'] / s['n_oos']:.0%}" in text.split("Correct refusals")[1].splitlines()[0]


def make_results(tmp_path):
    summary = {m: {"overall": {"n": 47, "chunk_hit@1": 0.5, "chunk_hit@5": 0.8, "chunk_mrr": 0.6, "page_hit@1": 0.7,
                               "page_hit@5": 0.95, "page_mrr": 0.8}, "by_type": {}, "comparison": {"n": 8, "all@5": 0.25}}
               for m in ("dense", "hybrid + rerank")}
    (tmp_path / "retrieval-final.json").write_text(json.dumps({"summary": summary}))
    for tag in ("baseline", "rerank"):
        (tmp_path / f"generation-{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))


def test_report_contains_counts_and_all_methods(tmp_path):
    make_results(tmp_path)
    text = mr.build_report(tmp_path)
    assert "| dense | 0.50 | 0.80 | 0.600 | 0.70 | 0.95 | 0.800 | 0.25 |" in text
    assert "47 single-page questions, 8 comparison" in text
    assert "| hybrid + rerank | 2/4 (50%) | 3/4 (75%) | 2/3 (67%) | 1/4 (25%) | 1/2 (50%) | 1.7s | 833 |" in text


def test_report_skips_missing_files(tmp_path):
    (tmp_path / "generation-baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    text = mr.build_report(tmp_path)
    assert "Answer quality" in text and "Retrieval" not in text and "hybrid + rerank" not in text
    with pytest.raises(SystemExit):
        mr.build_report(tmp_path / "empty-dir-does-not-exist")


def test_update_readme_replaces_only_the_block():
    readme = f"intro\n{mr.START}\nold table\n{mr.END}\noutro"
    out = mr.update_readme(readme, "NEW TABLE")
    assert out == f"intro\n{mr.START}\nNEW TABLE\n{mr.END}\noutro"
    assert mr.update_readme(out, "NEWER").count("NEWER") == 1       # idempotent: no duplicate blocks
    with pytest.raises(ValueError):
        mr.update_readme("no markers here", "x")
