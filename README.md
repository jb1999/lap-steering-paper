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

### SAE validation environment (separate)

The SAE-feature experiments (`scripts/sae_validation/`, `run_multi_direction_test.py`)
require `sae_lens` + `transformer_lens`, which conflict with the main repo's
pins on `torch` and `transformers`. Install them in a **separate** venv:

```bash
# create an isolated venv (any path; example uses ~/venvs/lap-sae)
uv venv ~/venvs/lap-sae --python 3.11

# install the SAE deps directly (this works as-is; do NOT use
# `uv pip install -r pyproject-sae.toml` — uv requires the file be named
# pyproject.toml or pylock.toml, so the shorthand fails)
VIRTUAL_ENV=~/venvs/lap-sae uv pip install \
    sae-lens==6.42.0 transformer-lens==3.0.0 \
    "torch>=2.5" "transformers>=4.50,<4.58" \
    datasets accelerate scikit-learn scipy "numpy<2" \
    tqdm safetensors einops "huggingface-hub<1.0" \
    sentencepiece protobuf
```

`pyproject-sae.toml` is committed as the human-readable source of truth for
these versions. To regenerate a lockfile from it via uv, copy it to a temp
subdir first (uv only auto-discovers `pyproject.toml`):

```bash
mkdir -p .sae-env && cp pyproject-sae.toml .sae-env/pyproject.toml
uv pip compile .sae-env/pyproject.toml -o requirements-sae.txt
VIRTUAL_ENV=~/venvs/lap-sae uv pip install -r requirements-sae.txt
```

**Running the SAE scripts.** Activate the venv first:

```bash
source ~/venvs/lap-sae/bin/activate

# Per-feature output-alignment vs activation comparison (App. G)
python scripts/sae_validation/02_sae_steering.py \
    --model google/gemma-2-2b --layer 22 \
    --prompts results/sae_validation/geography_prompts.json --top-k 20 \
    --multi-ks 1 5 20 100 --max-other 200 \
    --out sae_geography_l22_gemma2b.json

# Head-to-head with Arad et al. (App. arad_h2h): C_t(v) vs S_out and
# S_out^target across two intervention regimes. Runs all 4 model panels x
# 2 targets = 8 result JSONs (~540-555 features each, deterministic, seed=42).
python scripts/sae_validation/03_arad_head_to_head.py \
    --model google/gemma-2-2b --layer 22 \
    --sae-release gemma-scope-2b-pt-res-canonical --sae-width 16k \
    --prompts results/sae_validation/geography_prompts.json \
    --out arad_h2h_geography_l22_gemma2b.json
```

Some hosts need the cu13 .so files on `LD_LIBRARY_PATH`. The script auto-sets
this if it finds `~/venvs/lap-sae/lib/python3.11/site-packages/nvidia/cu13/lib`;
override with `LAP_SAE_CU13_LIB=/path/to/cu13/lib` (or set it to empty to
disable).

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

`check_results.py` re-derives ~110 headline statistics from `results/*.json`
across 10 sections (per-family $\alin$/$\amlp$, per-family steering, cross-concept
correlations, scaling, replication, multi-token demos, probe-derived steering
baseline, top-$k$ PCA control, SAE feature steering, Arad H2H) and diffs them against
the published baseline with stochastic-tolerant thresholds. Numbers reported
as MISSING indicate JSONs that haven't been produced on the current machine
yet (e.g. SAE results require the `lap-sae` venv). Use `--strict` to halve
the tolerances. See [`RESULTS.md`](RESULTS.md) for the full reference table.

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
| §4.2 — Probe-derived steering baseline (n=25) | `run_probe_steering_baseline.py` + `merge_probe_baseline.py` | `results/probe_steering/` |
| §4.4 — SAE feature steering (3 SAE releases)  | `scripts/sae_validation/02_sae_steering.py` | `results/sae_validation/` |
| §4.4 — Multi-direction PCA control            | `run_multi_direction_test.py` | `results/multi_direction/` |
| App. arad_h2h — Head-to-head with Arad et al. (8 panels) | `scripts/sae_validation/03_arad_head_to_head.py` | `results/sae_validation/arad_h2h_*.json` |
| Appendix — Failure-mode clustering            | `run_failure_modes.py` | `results/exp3/` |
| Appendix — Perturbation sensitivity (λ)       | `run_perturbation.py` | `results/exp4/` |
| Appendix — Refusal direction demo             | `run_refusal_demo.py` | `results/refusal_demo/` |
| Appendix — Refusal probe vs. mean-difference comparison | `run_refusal_probe_comparison.py` | `results/refusal_demo/probe_readout_comparison.json` |

The probe-baseline experiment is the n=25 extension that backs the
"separability ≠ steerability" claim in App. F. Run it as three parallel shards
to fit on multiple GPUs (~3–4 GPU-h per shard); CLI for the split is in
`probe_controlled_{A,B,C}.log`. Then run `scripts/merge_probe_baseline.py`
to produce the merged JSON and the headline numbers.

The SAE feature-steering experiments and the multi-direction PCA control use
the **separate `lap-sae` venv** described above, not the main repo venv.

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
