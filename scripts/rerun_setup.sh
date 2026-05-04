#!/usr/bin/env bash
# One-shot setup before kicking off the three rerun servers.
#
# 1. Backs up the existing results/ directory to results_pre_fix/ so the
#    pre-patch numbers are preserved for side-by-side comparison after the
#    rerun completes.
# 2. Removes the run_all checkpoint (if any) so a fresh re-run isn't skipped.
#
# This is a single-host operation — run it on any one server (the data dir
# is shared across all three).
#
# Usage:  bash scripts/rerun_setup.sh

set -euo pipefail

REPO="$HOME/adata/lap-steering-paper"
cd "$REPO"

if [[ -d results_pre_fix ]]; then
    echo "results_pre_fix/ already exists — assuming setup was run earlier."
    echo "If you want to redo the backup, delete results_pre_fix/ first."
    exit 1
fi

if [[ ! -d results ]]; then
    echo "No existing results/ directory found — nothing to back up."
    mkdir -p results
    exit 0
fi

echo "Backing up results/ -> results_pre_fix/ ..."
mv results results_pre_fix
mkdir -p results

# Drop the run_all checkpoint (if it migrated with results_pre_fix, leave it
# there for the historical record but don't let it influence the fresh run).
if [[ -f results_pre_fix/.run_all_checkpoint.json ]]; then
    echo "Pre-fix run_all checkpoint preserved at results_pre_fix/.run_all_checkpoint.json"
fi

echo
echo "Backup complete."
echo "  Old results: $REPO/results_pre_fix/"
echo "  Fresh dir:   $REPO/results/"
echo
echo "Cached activations from the pre-patch run live at:"
echo "  $REPO/results_pre_fix/exp1/gpu_cache/"
echo "(Server 1's run_steering.py will re-extract via the patched extractor"
echo "since results/exp1/gpu_cache/ no longer exists.)"
echo
echo "You can now kick off the three server scripts:"
echo "  Server 1 (~10–11h): bash scripts/rerun_server1_gemma.sh        2>&1 | tee server1.log"
echo "  Server 2 (~10–12h): bash scripts/rerun_server2_largemodels.sh  2>&1 | tee server2.log"
echo "  Server 3 (~9–11h):  bash scripts/rerun_server3_scaling.sh      2>&1 | tee server3.log"
