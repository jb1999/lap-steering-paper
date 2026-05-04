# Changelog

## Unreleased

### Data-pipeline fixes (re-runs)

- **Position-ids fix for left-padded RoPE models.** `ActivationExtractor`
  now passes correct `position_ids` when prompts are left-padded into a
  batch. Without this, several controlled families (notably parity, gender,
  geography on smaller models) reported attenuated steering magnitudes.
  All cross-concept, scaling, and replication results were re-run; expected
  values in `scripts/check_results.py` updated accordingly.
- **Chat-template application for refusal demo.** `run_refusal_demo.py` now
  applies the model's chat template to harmful/benign prompts (required for
  Llama-3.2-1B-Instruct to behave like an instruct model). Re-run produces
  the canonical 16-layer probe-accuracy and ΔP(comply) curves.

### New experiments

- **Arad et al. (2025) head-to-head** —
  `scripts/sae_validation/03_arad_head_to_head.py`. Computes our static
  output-alignment score $C_t(v)$, the Arad as-defined causal score
  $S_{\text{out}}$, and a target-conditioned variant $S_{\text{out}}^{\text{target}}$
  on 4 (model, layer) panels including the Arad published variant
  Gemma-2-9B-it/L31, and measures ground-truth ΔP under two intervention
  regimes (ours and Arad's amplification). 8 result JSONs in
  `results/sae_validation/arad_h2h_*.json`. Seed=42, deterministic.
- **Probe-derived steering baseline (n=25)** —
  `scripts/run_probe_steering_baseline.py` + `merge_probe_baseline.py`.
  Strong-baseline test of "separability ≠ steerability" using a trained
  probe's weight vector as the steering direction at every layer. Run
  across all 25 controlled families on Gemma-2-2B; output in
  `results/probe_steering/`.
- **Multi-direction PCA control** — `scripts/run_multi_direction_test.py`.
  Variance-aligned (not output-aligned) top-k PCA composition baseline
  against the multi-feature SAE result on geography and parity. Output in
  `results/multi_direction/`.
- **Refusal probe vs. mean-difference comparison** —
  `scripts/run_refusal_probe_comparison.py`. Side-by-side probe-derived
  vs. mean-difference steering on the refusal contrast across all 16
  layers. Output in `results/refusal_demo/probe_readout_comparison.json`.

### Tooling and validation

- `scripts/check_results.py` extended from 95 to 111 checks, adding the
  Arad H2H section (16 cells: per-panel `rho_cv_ours` and
  `rho_soutgt_arad`). Tight tolerances (±0.02–0.05) reflect the
  deterministic seed.
- New helper scripts added for parallel-server orchestration of the
  re-runs: `scripts/rerun_setup.sh` (one-time backup of pre-fix results)
  and `scripts/rerun_server{1,2,3}_*.sh` (sharded re-runs across three
  hosts).

### Environment

- `pyproject-sae.toml` committed as the human-readable dependency spec for
  the separate `lap-sae` virtualenv (sae_lens + transformer_lens conflict
  with the main repo's torch/transformers pins). Setup instructions in
  README.

## v1

Initial reproduction code release.
