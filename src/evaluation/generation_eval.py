"""Measure answer quality with an LLM judge. Costs API tokens, so it is resumable and sampled.

    python -m src.evaluation.generation_eval --n 20 --tag baseline
    python -m src.evaluation.generation_eval --tag baseline --summary-only

What is measured
  faithfulness      every claim in the answer is supported by the retrieved passages (judge)
  correctness       answer matches the reference answer: correct / partial / incorrect (judge)
  false refusal     answerable question, but the system said "I couldn't find this"
  correct refusal   out-of-scope question that was refused (no judge needed)

Limits: the judge is an LLM (a smaller one than the answerer). Treat scores as a consistent yardstick for
comparing versions, and read the flagged failures yourself, not as an absolute truth.
"""
import argparse
import json
import random
import time
from pathlib import Path

from src.config import JUDGE_MODEL, RESULTS_DIR
from src.evaluation.dataset import load_testset
from src.evaluation.make_testset import parse_json
from src.evaluation.metrics import mean
from src.generation.prompt import build_context

JUDGE_SYSTEM = """You are a strict evaluator of a documentation question-answering system.
You receive: retrieved passages, a question, a reference answer written from the source documentation, and the system's answer.
Evaluate two things independently.
1. faithful: true only if EVERY factual claim in the system's answer is supported by the retrieved passages. A claim that is true in general but is not stated in the passages counts as unsupported. List each unsupported claim.
2. verdict: "correct" if the answer contains the key facts of the reference answer and contradicts nothing in it; "partial" if it contains only some of the key facts; "incorrect" otherwise.
Reply with JSON only: {"faithful": true or false, "unsupported_claims": ["..."], "verdict": "correct" or "partial" or "incorrect", "reason": "one sentence"}"""

TYPE_ORDER = ["concept", "howto", "identifier", "comparison", "out_of_scope"]


def sample(testset: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Round-robin over question types so a small sample still covers everything."""
    rng = random.Random(seed)
    buckets = {t: [r for r in testset if r["type"] == t] for t in TYPE_ORDER}
    for b in buckets.values():
        rng.shuffle(b)
    out = []
    while len(out) < min(n, len(testset)):
        progressed = False
        for t in TYPE_ORDER:
            if buckets[t] and len(out) < n:
                out.append(buckets[t].pop())
                progressed = True
        if not progressed:
            break
    return out


def judge(row: dict, result: dict, llm) -> dict:
    user = (f"Retrieved passages:\n\n{build_context(result['retrieved'])}\n\n"
            f"Question: {row['question']}\n\nReference answer: {row['reference_answer']}\n\n"
            f"System's answer: {result['answer']}")
    out = llm([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
    obj = parse_json(out["text"]) or {}
    verdict = obj.get("verdict") if obj.get("verdict") in ("correct", "partial", "incorrect") else None
    return {"faithful": obj.get("faithful") if isinstance(obj.get("faithful"), bool) else None,
            "unsupported_claims": obj.get("unsupported_claims") or [], "verdict": verdict,
            "reason": obj.get("reason", ""), "judge_tokens": (out.get("prompt_tokens") or 0) + (out.get("completion_tokens") or 0),
            "judge_parse_failed": verdict is None or obj.get("faithful") is None}


def run_one(row: dict, answer_fn, judge_llm) -> dict:
    res = answer_fn(row["question"])
    rec = {"id": row["id"], "type": row["type"], "question": row["question"], "answer": res["answer"],
           "found": res["found"], "sources": [s["url"] for s in res["sources"]],
           "seconds": round(res["llm"]["seconds"] + res.get("retrieval_seconds", 0), 2),
           "tokens": (res["llm"].get("prompt_tokens") or 0) + (res["llm"].get("completion_tokens") or 0)}
    if row["should_refuse"]:
        rec["correct_refusal"] = not res["found"]
    elif not res["found"]:
        rec.update(false_refusal=True, faithful=None, verdict="incorrect", unsupported_claims=[], reason="refused", judge_tokens=0)
    else:
        rec.update(false_refusal=False, **judge(row, res, judge_llm))
    return rec


def summarize(rows: list[dict]) -> str:
    answerable = [r for r in rows if "correct_refusal" not in r]
    answered = [r for r in answerable if not r["false_refusal"]]
    judged = [r for r in answered if r["faithful"] is not None]
    oos = [r for r in rows if "correct_refusal" in r]
    lines = [f"Questions evaluated: {len(rows)} ({len(answerable)} answerable, {len(oos)} out-of-scope)", ""]
    if answerable:
        lines += [
            f"Correct (judge)        : {mean([r['verdict'] == 'correct' for r in answerable]):.0%}   "
            f"(correct or partial: {mean([r['verdict'] in ('correct', 'partial') for r in answerable]):.0%})",
            f"Faithful (judge)       : {mean([r['faithful'] for r in judged]):.0%}  of {len(judged)} answered questions",
            f"False refusals         : {mean([r['false_refusal'] for r in answerable]):.0%}  of {len(answerable)} answerable questions",
        ]
    if oos:
        lines.append(f"Correct refusals       : {mean([r['correct_refusal'] for r in oos]):.0%}  of {len(oos)} out-of-scope questions")
    lines.append(f"Mean latency / tokens  : {mean([r['seconds'] for r in rows]):.1f}s, {mean([r['tokens'] for r in rows]):.0f} tokens per answer")
    parse_fail = [r for r in answered if r.get("judge_parse_failed")]
    if parse_fail:
        lines.append(f"(judge reply could not be parsed for {len(parse_fail)} questions: re-run or inspect)")
    by_type = {}
    for t in TYPE_ORDER:
        sub = [r for r in answerable if r["type"] == t]
        if sub:
            by_type[t] = f"{t}: correct {mean([r['verdict'] == 'correct' for r in sub]):.0%}, faithful {mean([r['faithful'] for r in sub if r['faithful'] is not None]):.0%} (n={len(sub)})"
    if by_type:
        lines += ["", "By type:"] + [f"  {v}" for v in by_type.values()]
    unfaithful = [r for r in judged if r["faithful"] is False]
    if unfaithful:
        lines += ["", f"Unfaithful answers to read yourself ({len(unfaithful)}):"]
        for r in unfaithful[:8]:
            lines.append(f"  - {r['question']}\n      unsupported: {r['unsupported_claims'][:3]}")
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between questions (rate limits)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="answer with hybrid + cross-encoder reranking")
    args = ap.parse_args()

    path = RESULTS_DIR / f"generation-{args.tag}.jsonl"
    rows = load_rows(path)

    if not args.summary_only:
        from src.generation.llm import chat
        from src.generation.rag import answer
        from src.retrieval.hybrid import hybrid_search

        retriever = (lambda q, k: hybrid_search(q, k, rerank=True)) if args.rerank else None
        done = {r["id"] for r in rows}
        todo = [r for r in sample(load_testset(), args.n) if r["id"] not in done]
        print(f"{len(done)} already done, {len(todo)} to run (about {len(todo) * 4000:,} tokens). Judge model: {JUDGE_MODEL}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(todo, 1):
            rec = run_one(row, lambda q: answer(q, retriever=retriever),
                          lambda m: chat(m, model=JUDGE_MODEL, temperature=0, max_tokens=1200))
            with path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            print(f"  {i}/{len(todo)} [{rec['type']}] {rec.get('verdict', 'refused' if rec.get('correct_refusal') else 'ANSWERED OUT-OF-SCOPE')}  {row['question'][:60]}", flush=True)
            time.sleep(args.sleep)

    print("\n" + summarize(rows))
