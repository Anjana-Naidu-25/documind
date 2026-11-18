"""The full RAG pipeline: retrieve -> prompt -> generate -> attach sources.

    python -m src.generation.rag "how do I roll back a deployment?"
    python -m src.generation.rag "how do I roll back a deployment?" --debug
"""
import re
import sys
from typing import Callable

from src.config import TOP_K
from src.generation.prompt import NOT_FOUND, build_messages


def extract_citations(text: str, n_passages: int) -> list[int]:
    """Passage numbers the answer actually cites, ignoring numbers that don't exist."""
    return sorted({int(x) for x in re.findall(r"\[(\d+)\]", text) if 1 <= int(x) <= n_passages})


def is_not_found(text: str) -> bool:
    norm = text.replace("’", "'").lower()
    return NOT_FOUND.lower().rstrip(".") in norm


def answer(
    question: str,
    k: int = TOP_K,
    retriever: Callable | None = None,
    llm: Callable | None = None,
) -> dict:
    if retriever is None:
        from src.retrieval.hybrid import hybrid_search as retriever
    if llm is None:
        from src.generation.llm import chat as llm

    chunks = retriever(question, k)
    out = llm(build_messages(question, chunks))
    text = out["text"]

    not_found = is_not_found(text)
    cited = [] if not_found else extract_citations(text, len(chunks))
    sources = [{"n": n, "section": chunks[n - 1]["section"], "url": chunks[n - 1]["url"]} for n in cited]
    return {"question": question, "answer": text, "found": not not_found,
            "sources": sources, "retrieved": chunks, "llm": out}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    q = " ".join(args) or "how do I roll back a deployment?"
    res = answer(q)
    if "--debug" in sys.argv:
        print("Retrieved:")
        for i, c in enumerate(res["retrieved"], 1):
            print(f"  [{i}] dense#{c.get('dense_rank') or '-'} bm25#{c.get('bm25_rank') or '-'}  {c['section'][-90:]}")
        print()
    print(res["answer"])
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['section'][-80:]}\n      {s['url']}")
    o = res["llm"]
    print(f"\n({o['model']}, {o['prompt_tokens']} prompt + {o['completion_tokens']} completion tokens, {o['seconds']}s)")
