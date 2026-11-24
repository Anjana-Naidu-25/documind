# Evaluation results

Generated from `eval/results/` by `python -m src.evaluation.make_report`.

**Retrieval** (47 single-page questions, 8 comparison questions needing two pages). Chunk-level = the exact source chunk (strict); page-level = any chunk of the right page (lenient).

| Method | Chunk Hit@1 | Chunk Hit@5 | Chunk MRR | Page Hit@1 | Page Hit@5 | Page MRR | Both pages in top 5 |
|---|---|---|---|---|---|---|---|
| dense | 0.55 | 0.87 | 0.691 | 0.81 | 0.98 | 0.888 | 0.75 |
| bm25 | 0.60 | 0.89 | 0.735 | 0.79 | 0.96 | 0.872 | 0.25 |
| hybrid (RRF) | 0.60 | 0.89 | 0.735 | 0.79 | 1.00 | 0.885 | 0.38 |
| hybrid + rerank | 0.72 | 0.92 | 0.802 | 0.87 | 1.00 | 0.933 | 0.88 |

**Answer quality** (LLM judge; counts shown because the sample is small).

| Pipeline | Correct | Correct or partial | Faithful | False refusals | Out-of-scope refused | Mean latency | Tokens/answer |
|---|---|---|---|---|---|---|---|
| hybrid (RRF) | 16/16 (100%) | 16/16 (100%) | 16/16 (100%) | 0/16 (0%) | 4/4 (100%) | 5.6s | 1606 |
| hybrid + rerank | 15/16 (94%) | 16/16 (100%) | 15/16 (94%) | 0/16 (0%) | 4/4 (100%) | 7.2s | 1592 |
| hybrid + rerank, prompt v2 | 15/16 (94%) | 16/16 (100%) | 15/16 (94%) | 0/16 (0%) | 4/4 (100%) | 7.8s | 1685 |
