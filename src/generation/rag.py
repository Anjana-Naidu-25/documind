"""The full RAG pipeline: retrieve -> prompt -> generate -> attach sources.

    python -m src.generation.rag "how do I roll back a deployment?"
    python -m src.generation.rag "how do I roll back a deployment?" --debug
"""
import re
import sys
import time
from typing import Callable

from src.config import TOP_K
from src.generation.prompt import NOT_FOUND, build_messages

# Models sometimes write citations as 【1】 or [1, 2] instead of [1].
_CITE = re.compile(r"[\[\u3010]\s*(\d+(?:\s*[,;]\s*\d+)*)\s*[\]\u3011]")
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def normalize_citations(text: str) -> str:
    """Show every citation in one style: 【1】 -> [1]."""
    return re.sub(r"\u3010\s*(\d+)\s*\u3011", r"[\1]", text)


def extract_citations(text: str, n_passages: int) -> list[int]:
    """Passage numbers the answer actually cites, ignoring numbers that don't exist.
    Code blocks are skipped so `items[1]` in a command is not mistaken for a citation."""
    found: set[int] = set()
    for m in _CITE.finditer(_CODE_FENCE.sub("", text)):
        for part in re.split(r"[,;]", m.group(1)):
            n = int(part)
            if 1 <= n <= n_passages:
                found.add(n)
    return sorted(found)


def is_not_found(text: str) -> bool:
    norm = text.replace("’", "'").lower()
    return NOT_FOUND.lower().rstrip(".") in norm


def answer(
    question: str,
    k: int = TOP_K,
    retriever: Callable | None = None,
    llm: Callable | None = None,
    rerank: bool = False,
) -> dict:
    if retriever is None:
        from src.retrieval.hybrid import hybrid_search
        retriever = lambda q, n: hybrid_search(q, n, rerank=rerank)
    if llm is None:
        from src.generation.llm import chat as llm

    t0 = time.time()
    chunks = retriever(question, k)
    retrieval_seconds = round(time.time() - t0, 2)
    out = llm(build_messages(question, chunks))
    text = normalize_citations(out["text"])

    not_found = is_not_found(text)
    cited = [] if not_found else extract_citations(text, len(chunks))
    sources = [{"n": n, "section": chunks[n - 1]["section"], "url": chunks[n - 1]["url"]} for n in cited]
    return {"question": question, "answer": text, "found": not not_found,
            "sources": sources, "retrieved": chunks, "llm": out,
            "retrieval_seconds": retrieval_seconds}


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
