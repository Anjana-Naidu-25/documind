"""Turn eval/results/* into Markdown tables, so README numbers come straight from the result files.

    python -m src.evaluation.make_report

Writes docs/RESULTS.md and replaces the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""
import json
import re
from pathlib import Path

from src.config import RESULTS_DIR, ROOT
from src.evaluation.generation_eval import load_rows, stats

RETRIEVAL_FILE = "retrieval-final.json"
# All three runs use the same 20 questions (same seed), so they are directly comparable. Missing files are skipped.
GENERATION_RUNS = [("hybrid (RRF)", "baseline-now"), ("hybrid + rerank", "rerank-now"), ("hybrid + rerank, prompt v2", "rerank-v2")]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _frac(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.0%})" if d else "n/a"


def retrieval_table(data: dict) -> str:
    summary = data["summary"]
    first = next(iter(summary.values()))
    n_single, n_cmp = first["overall"]["n"], first["comparison"]["n"]
    lines = [
        f"**Retrieval** ({n_single} single-page questions, {n_cmp} comparison questions needing two pages). "
        "Chunk-level = the exact source chunk (strict); page-level = any chunk of the right page (lenient).",
        "",
        "| Method | Chunk Hit@1 | Chunk Hit@5 | Chunk MRR | Page Hit@1 | Page Hit@5 | Page MRR | Both pages in top 5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        o, c = m["overall"], m["comparison"]
        lines.append(f"| {name} | {o['chunk_hit@1']:.2f} | {o['chunk_hit@5']:.2f} | {o['chunk_mrr']:.3f} | "
                     f"{o['page_hit@1']:.2f} | {o['page_hit@5']:.2f} | {o['page_mrr']:.3f} | {c['all@5']:.2f} |")
    return "\n".join(lines)


def answer_table(runs: dict[str, list[dict]]) -> str:
    lines = ["**Answer quality** (LLM judge; counts shown because the sample is small).", "",
             "| Pipeline | Correct | Correct or partial | Faithful | False refusals | Out-of-scope refused | Mean latency | Tokens/answer |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rows in runs.items():
        s = stats(rows)
        lines.append(f"| {name} | {_frac(s['n_correct'], s['n_answerable'])} | "
                     f"{_frac(s['n_correct'] + s['n_partial'], s['n_answerable'])} | {_frac(s['n_faithful'], s['n_judged'])} | "
                     f"{_frac(s['n_false_refusal'], s['n_answerable'])} | {_frac(s['n_correct_refusal'], s['n_oos'])} | "
                     f"{s['mean_seconds']:.1f}s | {s['mean_tokens']:.0f} |")
    return "\n".join(lines)


def build_report(results_dir: Path = RESULTS_DIR) -> str:
    parts = []
    path = results_dir / RETRIEVAL_FILE
    if path.exists():
        parts.append(retrieval_table(json.loads(path.read_text())))
    runs = {name: load_rows(results_dir / f"generation-{tag}.jsonl") for name, tag in GENERATION_RUNS}
    runs = {k: v for k, v in runs.items() if v}
    if runs:
        parts.append(answer_table(runs))
    if not parts:
        raise SystemExit(f"No result files found in {results_dir}")
    return "\n\n".join(parts)


def update_readme(readme: str, block: str) -> str:
    if START not in readme or END not in readme:
        raise ValueError(f"README is missing the {START} ... {END} markers")
    return re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", readme, flags=re.DOTALL)


if __name__ == "__main__":
    block = build_report()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "RESULTS.md").write_text("# Evaluation results\n\nGenerated from `eval/results/` by `python -m src.evaluation.make_report`.\n\n" + block + "\n")
    readme_path = ROOT / "README.md"
    readme_path.write_text(update_readme(readme_path.read_text(), block))
    print(block)
    print("\nWrote docs/RESULTS.md and updated README.md")
