#!/usr/bin/env bash
# Server 2 — Large-model replication (~10–12 GPU-hours on a 3090 Ti)
#
# Per-layer A_lin extraction on the three 7–8B replication models, plus the
# cross-concept steerability run on Llama-8B (the most expensive single sweep).
#
# Usage:  bash scripts/rerun_server2_largemodels.sh 2>&1 | tee server2.log

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

# Per-layer A_lin replication on the three 7–8B transformers
run "$PY" scripts/run_replication.py --model meta-llama/Llama-3.1-8B  --device cuda
run "$PY" scripts/run_replication.py --model mistralai/Mistral-7B-v0.3 --device cuda
run "$PY" scripts/run_replication.py --model Qwen/Qwen2.5-7B          --device cuda

# Cross-concept steerability on Llama-8B (the most expensive single sweep)
run "$PY" scripts/run_steerability.py --model meta-llama/Llama-3.1-8B --device cuda

echo; echo "===== Server 2 complete ====="; date
