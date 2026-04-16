# Predicting Where Steering Vectors Succeed

Reproduction code for the paper *Predicting Where Steering Vectors Succeed*
(Billa, 2026). Repo: <https://github.com/jb1999/lap-steering-paper>

The Linear Accessibility Profile (LAP) uses the **logit lens** to predict
whether difference-of-means steering will work for a given concept and at which
layer. This repo reproduces every experiment in the paper across 10+ models
(Gemma, Pythia 160M–6.9B, Qwen, Llama, Mistral, OLMo, Mamba, RWKV) and 25+
concept families.

## Setup

```bash
# uv (recommended — pinned to torch 2.5 / transformers <4.51)
uv sync

# or pip
pip install -e .
```

If you keep multiple virtualenvs, set `UV_PROJECT_ENVIRONMENT=/path/to/venv`
to keep `uv` from creating a new `.venv` in this repo.

**Hardware:** single NVIDIA GPU with ≥24 GB VRAM (tested on RTX 3090 Ti). All
models fit in VRAM individually; the largest is Llama-3.1-8B at ~16 GB.

## Quick start

```bash
# Sanity-check the pipeline (2 min)
python scripts/validate.py --device cuda

# Run everything (~6 GPU-hours; resumable via results/.run_all_checkpoint.json)
python scripts/run_all.py --device cuda

# After the run, verify your numbers match the paper:
python scripts/check_results.py
```

`check_results.py` re-derives 66 headline statistics from `results/*.json` and
diffs them against the published baseline with stochastic-tolerant thresholds.
A clean run prints `Summary: 66/66 checks passed`. Use `--strict` to halve the
tolerances. See [`RESULTS.md`](RESULTS.md) for the full reference table.

## Reproducing the paper piecewise

Each script is independently runnable. Standard flags: `--device cuda`,
`--model <hf-id>`, `--results-dir results`.

| Paper section | Script | Outputs |
|---|---|---|
| §3.1, §4.1 — A_lin / A_mlp emergence (Gemma) | `run_emergence.py` | `results/exp1/` |
| §4.2 — Steering correlation across layers     | `run_steering.py`  | `results/exp2/` |
| §4.2 — Cross-concept steerability (5 models)  | `run_steerability.py` | `results/cross_concept/` |
| §4.2 — Pythia scaling 160M–6.9B               | `run_scaling.py`   | `results/scaling/` |
| §4.2 — C(d), KL, steering efficiency          | `run_theory.py`    | `results/theory/` |
| §4.3 — Cross-architecture replication         | `run_replication.py` | `results/replication_*/` |
| §4.3 — SSM replication (Mamba, RWKV)          | `run_replication_ssm.py` | `results/replication_*/` |
| §4.3 — London→Paris entity steering           | `run_entity_steering.py` | `results/entity_steering_*/` |
| Appendix — Failure-mode clustering            | `run_failure_modes.py` | `results/exp3/` |
| Appendix — Perturbation sensitivity (λ)       | `run_perturbation.py` | `results/exp4/` |
| Appendix — Refusal direction demo             | `run_refusal_demo.py` | `results/refusal_demo/` |

## Reference results in `results/`

The result JSONs are committed (~1 MB total) so users can:

1. Diff their own run against the paper's run with `check_results.py`.
2. Reanalyse without re-running the model sweeps.

The 600 MB raw activation cache (`results/exp1/gpu_cache/`) and the run-time
log files are gitignored — they're regenerable from the scripts.

## Tooling

- `scripts/check_results.py` — automated PASS/FAIL diff against paper baseline.
- `scripts/compute_paper_stats.py` — reprints all paper-table numbers from JSONs.
- `scripts/generate_paper_csvs.py` — emits the CSVs consumed by the paper's pgfplots figures (default `--out paper_csvs/`).
- `scripts/generate_figures.py` — sanity-check PNG/PDF previews of each figure (the paper sources data from CSV via pgfplots, so these are not used in the LaTeX build).

## Project layout

```
src/
  data/         core families + 25 controlled binary families + collateral set
  extraction/   activation extractor with hook-based perturbation injection
  probes/       LinearProbe (logit lens) and MLPProbe with 5-fold CV
  metrics/      A_lin, A_mlp, gap, λ, effective rank, condition number
  steering/     difference-of-means steering pipeline + KL collateral
  analysis/     failure clustering, cross-concept prediction, figures
scripts/        per-experiment runners + check / report helpers
configs/        default experiment configuration
results/        committed reference outputs (one subdir per experiment)
```

## Citation

```bibtex
@article{billa2026predicting,
  title={Predicting Where Steering Vectors Succeed},
  author={Billa, Jayadev},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
