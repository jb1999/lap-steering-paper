# Reference Results

This file documents the headline numbers produced by the regression run that
backs the paper "Predicting Where Steering Vectors Succeed". The corresponding
JSON files are committed under `results/` so a fresh run can be diffed against
them.

To verify your run automatically, use:

```bash
python scripts/check_results.py           # tolerances ~0.05 abs (~3 sigma)
python scripts/check_results.py --strict  # half-tolerances
```

All numbers below are produced on a single RTX 3090 Ti with the dependency pin
in `pyproject.toml`. Tolerances exist because logistic-regression CV folds and
KL collateral sampling are non-deterministic; magnitudes and signs are stable.

## Experiment 1 / 2 — Per-family on Gemma-2-2B (paper Tables 2, 3)

| Family | Model acc | peak A_lin (L) | peak A_mlp (L) | gap | max ΔP (L) | rho(A_lin, ΔP) |
|---|---:|---:|---:|---:|---:|---:|
| arithmetic     | 0.657 | 0.686 (23) | 0.904 (22) | 0.218 | 0.246 (24) | +0.719 |
| geography      | 0.601 | 0.280 (25) | 1.000 (23) | 0.720 | 0.238 (23) | +0.758 |
| sequence       | 0.778 | 0.711 (24) | 0.954 (21) | 0.243 | 0.157 (24) | +0.814 |
| word_transform | 0.623 | 0.512 (24) | 0.647 (25) | 0.135 | 0.386 (24) | +0.871 |
| analogy        | 0.716 | 0.451 (24) | 0.629 (24) | 0.177 | 0.191 (24) | +0.866 |

**Pooled** (n=130 layer×family pairs): rho(A_lin, ΔP) = **+0.777**, partial
r controlling for layer = **+0.507** (both p < 1e-9).

## Cross-concept steerability (paper Table 7)

Spearman rho between peak A_lin and max ΔP across controlled binary families.

| Subset            | Pythia-2.8B | Gemma-2-2B | Qwen-1.5B | Qwen-7B | Llama-8B |
|---|---:|---:|---:|---:|---:|
| All families      | +0.89 | +0.86 | +0.90 | +0.92 | +0.93 |
| Controlled (n=23–24) | +0.86 | +0.86 | +0.86 | +0.90 | +0.91 |
| Ctrl A_lin > 0.05 | +0.50 | +0.68 | +0.86 | +0.90 | +0.83 |
| Ctrl A_lin > 0.10 | +0.50 | +0.52 | +0.93 | +0.86 | +0.84 |

## Scaling — Pythia 160M to 6.9B (paper Table 8)

| Size  | n  | mean A_lin | mean ΔP | steerable | rho(A_lin, ΔP) |
|---|---:|---:|---:|---:|---:|
| 160M  | 23 | 0.055 | 0.011 | 5/23 | +0.42 |
| 410M  | 23 | 0.075 | 0.038 | 6/23 | +0.71 |
| 1B    | 23 | 0.087 | 0.022 | 7/23 | +0.68 |
| 2.8B  | 23 | 0.111 | 0.030 | 9/23 | +0.86 |
| 6.9B  | 23 | 0.118 | 0.037 | 9/23 | +0.86 |

Two of the 25 controlled families (`c_animal`, `c_edible`) are dropped on
Pythia because their target tokens (`mammal`, `inedible`) tokenize to multiple
tokens, preventing single-token steering evaluation.

## Cross-architecture replication (paper Table 4)

Peak A_lin (layer) and per-family Spearman rho on the 5 core families.

| Family         | Llama-8B (32L)  | Mistral-7B (32L) | Qwen-7B (28L)   |
|---|---:|---:|---:|
| arithmetic     | 0.995 (29) +0.85 | 0.695 (30) +0.89 | 0.935 (26) +0.71 |
| geography      | 0.680 (25) +0.93 | 0.540 (25) +0.79 | 0.585 (25) +0.72 |
| sequence       | 0.820 (31) +0.90 | 0.735 (31) +0.89 | 0.780 (26) +0.66 |
| word_transform | 0.715 (31)  —    | 0.650 (31)  —    | 0.500 (26)  —    |
| analogy        | 0.415 (29)  —    | 0.765 (31)  —    | 0.385 (26)  —    |

Word-transform and analogy steering targets too sparse (n<10) for stable rho;
A_lin emergence still tracks model accuracy.

Mistral requires the SentencePiece `▁`-prefix fallback in `get_token_id`
(`scripts/run_replication.py` lines 24–34), without which digit targets are
multi-token and accuracy is 0.

## Demos

| Demo                              | rho(A_lin, ΔP) | p     |
|---|---:|---:|
| Entity steering — Gemma-2-2B (London→Paris) | +0.660 | 2e-4 |
| Entity steering — OLMo-2-1B-Instruct        | +0.742 | 1e-3 |
| Refusal direction — Llama-3.2-1B-Instruct   | +0.908 | 2e-3 |

## What gets checked

`scripts/check_results.py` covers the 66 headline values above. Anything not
covered (per-layer trajectories, full controlled-family tables, theory C(d)
metrics, SSM replication, perturbation/chaotic results) is still recorded in
the result JSONs and can be inspected directly or re-derived via
`scripts/compute_paper_stats.py`.
