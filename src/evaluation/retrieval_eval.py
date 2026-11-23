"""Compare retrieval methods on the test set. No LLM calls, so it is free and fast.

    python -m src.evaluation.retrieval_eval                 # dense, bm25, hybrid
    python -m src.evaluation.retrieval_eval --rerank        # also hybrid + cross-encoder reranker
    python -m src.evaluation.retrieval_eval --misses 5      # show questions the best method missed

Chunk-level = the exact chunk the question was written from (strict).
Page-level  = any chunk of the right page (lenient: the answer may also live in neighbouring chunks).
"""
import argparse
import json
import time
from typing import Callable

from src.config import RESULTS_DIR
from src.evaluation.dataset import load_testset
from src.evaluation.metrics import coverage_at_k, dedupe_keep_order, hit_at_k, mean, reciprocal_rank

KS = (1, 3, 5, 10)


def evaluate(testset: list[dict], rankers: dict[str, Callable[[str], list[str]]], id2url: dict[str, str]) -> dict:
    """rankers: name -> function(question) -> ranked chunk ids (best first). Returns per-question and summary data."""
    per_q: dict[str, list[dict]] = {name: [] for name in rankers}
    for row in testset:
        if row["should_refuse"]:
            continue
        for name, rank in rankers.items():
            ids = rank(row["question"])
            urls = dedupe_keep_order([id2url[i] for i in ids])
            rel_urls, rel_chunks = set(row["expected_urls"]), set(row["expected_chunk_ids"])
            rec = {"id": row["id"], "type": row["type"], "question": row["question"], "top_urls": urls[:5]}
            if row["type"] == "comparison":
                rec.update({f"cov@{k}": coverage_at_k(urls, rel_urls, k) for k in (5, 10)})
                rec["all@5"] = 1.0 if coverage_at_k(urls, rel_urls, 5) == 1.0 else 0.0
            else:
                for k in KS:
                    rec[f"page_hit@{k}"] = hit_at_k(urls, rel_urls, k)
                    rec[f"chunk_hit@{k}"] = hit_at_k(ids, rel_chunks, k)
                rec["page_mrr"] = reciprocal_rank(urls, rel_urls)
                rec["chunk_mrr"] = reciprocal_rank(ids, rel_chunks)
            per_q[name].append(rec)

    def agg(rows: list[dict], keys: list[str]) -> dict:
        return {k: round(mean([r[k] for r in rows]), 3) for k in keys} | {"n": len(rows)}

    single_keys = [f"chunk_hit@{k}" for k in KS] + ["chunk_mrr"] + [f"page_hit@{k}" for k in KS] + ["page_mrr"]
    summary: dict = {}
    for name, rows in per_q.items():
        single = [r for r in rows if r["type"] != "comparison"]
        types = dict.fromkeys(r["type"] for r in single)
        summary[name] = {
            "overall": agg(single, single_keys),
            "by_type": {t: agg([r for r in single if r["type"] == t], single_keys) for t in types},
            "comparison": agg([r for r in rows if r["type"] == "comparison"], ["cov@5", "cov@10", "all@5"]),
        }
    return {"per_question": per_q, "summary": summary}


def format_report(summary: dict) -> str:
    names = list(summary)
    lines = []
    lines += ["### Chunk-level (strict): the exact source chunk", "",
              "| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |", "|---|---|---|---|---|---|"]
    for n in names:
        o = summary[n]["overall"]
        lines.append(f"| {n} | {o['chunk_hit@1']:.2f} | {o['chunk_hit@3']:.2f} | {o['chunk_hit@5']:.2f} | {o['chunk_hit@10']:.2f} | {o['chunk_mrr']:.3f} |")
    lines += ["", "### Page-level (lenient): any chunk of the right page", "",
              "| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |", "|---|---|---|---|---|---|"]
    for n in names:
        o = summary[n]["overall"]
        lines.append(f"| {n} | {o['page_hit@1']:.2f} | {o['page_hit@3']:.2f} | {o['page_hit@5']:.2f} | {o['page_hit@10']:.2f} | {o['page_mrr']:.3f} |")
    types = list(summary[names[0]]["by_type"])
    lines += ["", "### Page-level Hit@5 by question type", "", "| Method | " + " | ".join(f"{t} (n={summary[names[0]]['by_type'][t]['n']})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{summary[n]['by_type'][t]['page_hit@5']:.2f}" for t in types) + " |")
    c0 = summary[names[0]]["comparison"]
    if c0["n"]:
        lines += ["", f"### Comparison questions (need 2 pages, n={c0['n']})", "",
                  "| Method | Both pages in top 5 | Mean coverage@5 | Mean coverage@10 |", "|---|---|---|---|"]
        for n in names:
            c = summary[n]["comparison"]
            lines.append(f"| {n} | {c['all@5']:.2f} | {c['cov@5']:.2f} | {c['cov@10']:.2f} |")
    return "\n".join(lines)


def build_rankers(use_rerank: bool, id2rec: dict[str, dict]) -> dict[str, Callable[[str], list[str]]]:
    from src.retrieval.hybrid import _bm25_ids, _dense_ids, rrf
    cache: dict[str, tuple[list[str], list[str]]] = {}

    def both(q: str):                       # run each retriever once per question and share the lists
        if q not in cache:
            cache[q] = (_dense_ids(q, 20), _bm25_ids(q, 20))
        return cache[q]

    rankers = {
        "dense": lambda q: both(q)[0][:10],
        "bm25": lambda q: both(q)[1][:10],
        "hybrid (RRF)": lambda q: [i for i, _ in rrf(list(both(q)))][:10],
    }
    if use_rerank:
        from src.retrieval.rerank import rerank

        def hybrid_rerank(q: str) -> list[str]:
            cands = [id2rec[i] for i, _ in rrf(list(both(q)))[:20]]
            return [r["id"] for r in rerank(q, cands, top_k=10)]
        rankers["hybrid + rerank"] = hybrid_rerank
    return rankers


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--misses", type=int, default=0, help="print N questions the last method missed at page-level top 5")
    args = ap.parse_args()

    from src.retrieval.bm25 import get_index
    id2rec = {r["id"]: r for r in get_index().records}
    id2url = {i: r["url"] for i, r in id2rec.items()}
    testset = load_testset()

    start = time.time()
    out = evaluate(testset, build_rankers(args.rerank, id2rec), id2url)
    print(format_report(out["summary"]))
    print(f"\n({sum(not r['should_refuse'] for r in testset)} answerable questions, {time.time() - start:.0f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"retrieval-{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")

    if args.misses:
        last = list(out["per_question"])[-1]
        misses = [r for r in out["per_question"][last] if r.get("page_hit@5") == 0.0][:args.misses]
        print(f"\nMisses for '{last}':")
        for r in misses:
            print(f"- [{r['type']}] {r['question']}\n    got: {r['top_urls'][:3]}")
