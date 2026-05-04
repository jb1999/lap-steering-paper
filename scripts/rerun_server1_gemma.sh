#!/usr/bin/env bash
# Server 1 — Gemma-2-2B primary experiments (~10–11 GPU-hours on a 3090 Ti)
#
# Includes the validation pass + everything the paper does on Gemma-2-2B as the
# primary model: emergence, within-concept steering, cross-concept steerability,
# C(d)/theory analysis, failure-mode clustering, perturbation sensitivity,
# and the two demos that use Gemma.
#
# After all servers finish, rsync results/ from servers 2 and 3 onto this host
# and run scripts/generate_figures.py to regenerate paper figures.
#
# Usage:  bash scripts/rerun_server1_gemma.sh 2>&1 | tee server1.log

set -euo pipefail

REPO="$HOME/adata/lap-steering-paper"
VENV="${LAP_VENV:-$HOME/venvs/lap-repro}"
PY="$VENV/bin/python"

cd "$REPO"

run() { echo; echo "===== $* ====="; date; "$@"; }

# Sanity check that the patches loaded
run "$PY" -c "from src.extraction.activations import ActivationExtractor; \
    import inspect; src = inspect.getsource(ActivationExtractor); \
    assert '_fix_position_ids' in src and src.count('self._fix_position_ids') == 5, \
    'Position-ids fix patches missing!'; print('Patch sanity OK')"

run "$PY" scripts/validate.py --device cuda

# Primary Gemma-2-2B experiments
run "$PY" scripts/run_emergence.py    --model google/gemma-2-2b --device cuda
run "$PY" scripts/run_steering.py     --model google/gemma-2-2b --device cuda
run "$PY" scripts/run_steerability.py --model google/gemma-2-2b --device cuda
run "$PY" scripts/run_theory.py       --model google/gemma-2-2b --device cuda
run "$PY" scripts/run_failure_modes.py --device cuda
run "$PY" scripts/run_perturbation.py --device cuda

# Demos that need Gemma
run "$PY" scripts/run_refusal_demo.py     --device cuda
run "$PY" scripts/run_entity_steering.py  --model google/gemma-2-2b --device cuda

echo; echo "===== Server 1 complete ====="; date
