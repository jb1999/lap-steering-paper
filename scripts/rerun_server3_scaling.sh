#!/usr/bin/env bash
# Server 3 — Scaling, mid-size steerability, SSMs, OLMo demo (~9–11 GPU-hours)
#
# Pythia scaling sweep, two more cross-concept steerability runs (Qwen-1.5B
# and Qwen-7B), the two non-transformer A_lin replications, and the OLMo
# entity-steering demo.
#
# Usage:  bash scripts/rerun_server3_scaling.sh 2>&1 | tee server3.log

set -euo pipefail

REPO="$HOME/adata/lap-steering-paper"
VENV="${LAP_VENV:-$HOME/venvs/lap-repro}"
PY="$VENV/bin/python"

cd "$REPO"

run() { echo; echo "===== $* ====="; date; "$@"; }

run "$PY" -c "from src.extraction.activations import ActivationExtractor; \
    import inspect; src = inspect.getsource(ActivationExtractor); \
    assert '_fix_position_ids' in src and src.count('self._fix_position_ids') == 5, \
    'Position-ids fix patches missing!'; print('Patch sanity OK')"

# Pythia scaling sweep (calls run_steerability internally for each size)
run "$PY" scripts/run_scaling.py --device cuda

# Cross-concept steerability on Qwen-1.5B and Qwen-7B
run "$PY" scripts/run_steerability.py --model Qwen/Qwen2.5-1.5B --device cuda
run "$PY" scripts/run_steerability.py --model Qwen/Qwen2.5-7B   --device cuda

# Non-transformer A_lin replication (no RoPE — fix is a no-op for these)
run "$PY" scripts/run_replication_ssm.py --model state-spaces/mamba-1.4b-hf --device cuda
run "$PY" scripts/run_replication_ssm.py --model RWKV/v6-Finch-1B6-HF      --device cuda

# OLMo entity-steering demo (separate output dir to avoid clobbering Gemma demo)
run "$PY" scripts/run_entity_steering.py \
    --model allenai/OLMo-2-0425-1B-Instruct \
    --results-dir results/entity_steering_olmo \
    --device cuda

echo; echo "===== Server 3 complete ====="; date
