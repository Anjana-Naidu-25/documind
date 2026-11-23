"""Build eval/testset.jsonl: LLM-written questions from real chunks + curated questions.

    python -m src.evaluation.make_testset --per-type 16 --seed 42

The generated questions are only a draft. Open eval/testset.jsonl and delete lines whose question is
ambiguous, answerable from several places, or just bad. Keep the file in Git.
"""
import argparse
import json
import random
import re
import time

from src.config import TESTSET_PATH
from src.evaluation.dataset import curated_rows, save_testset
from src.retrieval.bm25 import load_records

SKIP_SECTIONS = {"before you begin", "objectives", "cleaning up", "what's next", "see also", "lesson content"}
IDENT = re.compile(r"`(--[a-z-]+|[a-z]+[A-Z][A-Za-z]{3,}|kubectl [a-z-]+[^`]*)`")

TYPE_INSTRUCTIONS = {
    "concept": "Write ONE question a Kubernetes learner might ask about a concept explained in the passage (for example 'What is ...?' or 'Why does ...?').",
    "howto": "Write ONE practical how-to question (for example 'How do I ...?') whose answer is given in the passage.",
    "identifier": "Write ONE question that names a specific field, flag, command or resource that appears in the passage, and asks what it does or how to use it.",
}

GEN_SYSTEM = """You write evaluation questions for a Kubernetes documentation search system.
You are given one passage from the documentation. {instruction}
Rules:
- The question must be fully understandable on its own, so name the Kubernetes concept explicitly (never say "this passage" or "the above").
- Do not copy phrases from the passage. Paraphrase, the way a user who has not read the docs would ask.
- The passage must contain enough information to answer it.
- Also write a reference answer of 1-3 sentences that uses only facts from the passage.
Reply with JSON only: {{"question": "...", "reference_answer": "..."}}"""


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a model reply (models sometimes add prose or code fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _tokens(r: dict) -> int:
    return r.get("tokens") or int(len(r["text"].split()) * 1.3)


def candidate_pool(records: list[dict]) -> dict[str, list[dict]]:
    """Group usable chunks by question type."""
    pool = {"concept": [], "howto": [], "identifier": []}
    for r in records:
        if _tokens(r) < 100 or r["section"].split(" > ")[-1].strip().lower() in SKIP_SECTIONS:
            continue
        has_code = "```" in r["text"]
        if IDENT.search(r["text"]):
            pool["identifier"].append(r)
        if has_code:
            pool["howto"].append(r)
        elif not IDENT.search(r["text"]):
            pool["concept"].append(r)
    return pool


def select_chunks(records: list[dict], per_type: int, seed: int) -> list[tuple[str, dict]]:
    """Pick per_type chunks for each type, at most one per page, no page used twice overall."""
    rng = random.Random(seed)
    pool = candidate_pool(records)
    used_pages: set[str] = set()
    picked: list[tuple[str, dict]] = []
    for qtype in ("identifier", "howto", "concept"):          # rarest first so they get enough pages
        candidates = pool[qtype][:]
        rng.shuffle(candidates)
        n = 0
        for r in candidates:
            if r["url"] in used_pages:
                continue
            used_pages.add(r["url"])
            picked.append((qtype, r))
            n += 1
            if n == per_type:
                break
    return picked


def generate_question(qtype: str, chunk: dict, llm) -> dict | None:
    messages = [
        {"role": "system", "content": GEN_SYSTEM.format(instruction=TYPE_INSTRUCTIONS[qtype])},
        {"role": "user", "content": f"Passage:\n\n{chunk['text']}"},
    ]
    obj = parse_json(llm(messages)["text"])
    if not obj or not obj.get("question") or not obj.get("reference_answer"):
        return None
    q = str(obj["question"]).strip()
    if not 15 <= len(q) <= 250:
        return None
    return {"question": q, "reference_answer": str(obj["reference_answer"]).strip()}


def build(records: list[dict], per_type: int, seed: int, llm, sleep: float = 0.5) -> list[dict]:
    rows = []
    for qtype, chunk in select_chunks(records, per_type, seed):
        g = generate_question(qtype, chunk, llm)
        if g is None:
            print(f"  skipped a bad generation for {chunk['url']}")
            continue
        rows.append({"id": f"{qtype[:3]}-{len(rows) + 1:02d}", "type": qtype, **g,
                     "expected_urls": [chunk["url"]], "expected_chunk_ids": [chunk["id"]], "should_refuse": False})
        print(f"  [{qtype}] {g['question']}")
        time.sleep(sleep)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing testset")
    args = ap.parse_args()

    if TESTSET_PATH.exists() and not args.force:
        raise SystemExit(f"{TESTSET_PATH} already exists. It may contain your manual edits; use --force to overwrite.")

    from src.generation.llm import chat
    rows = build(load_records(), args.per_type, args.seed, llm=lambda m: chat(m, max_tokens=600)) + curated_rows()
    save_testset(rows)
    by_type = {t: sum(r["type"] == t for r in rows) for t in dict.fromkeys(r["type"] for r in rows)}
    print(f"\nSaved {len(rows)} questions to {TESTSET_PATH}: {by_type}")
    print("Now open the file and delete any question that is ambiguous or bad.")
