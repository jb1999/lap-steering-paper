"""Run all experiments for the LAP steering paper.

Tracks progress in results/.run_all_checkpoint.json so that re-running
after a crash skips already-completed scripts.

Usage:
    python scripts/run_all.py --device cuda
    python scripts/run_all.py --device cuda --skip-replication
    python scripts/run_all.py --reset   # clear checkpoint, start fresh
"""

import scripts._env  # noqa: F401

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    ("Validation", "scripts/validate.py"),
    ("Emergence (Section 4.1)", "scripts/run_emergence.py"),
    ("Steering (Section 4.2)", "scripts/run_steering.py"),
    ("Steerability (Section 4.2)", "scripts/run_steerability.py"),
    ("Theory / C(d) analysis", "scripts/run_theory.py"),
    ("Failure modes (Appendix)", "scripts/run_failure_modes.py"),
    ("Perturbation sensitivity (Appendix)", "scripts/run_perturbation.py"),
    ("Scaling (Section 4.2)", "scripts/run_scaling.py"),
]

REPLICATION = [
    ("Replication: Llama-3.1-8B", "scripts/run_replication.py", ["--model", "meta-llama/Llama-3.1-8B"]),
    ("Replication: Mistral-7B", "scripts/run_replication.py", ["--model", "mistralai/Mistral-7B-v0.3"]),
    ("Replication: Qwen2.5-7B", "scripts/run_replication.py", ["--model", "Qwen/Qwen2.5-7B"]),
    ("Replication: Mamba-1.4B", "scripts/run_replication_ssm.py", ["--model", "state-spaces/mamba-1.4b-hf"]),
    ("Replication: RWKV-1.6B", "scripts/run_replication_ssm.py", ["--model", "RWKV/v6-Finch-1B6-HF"]),
    # Cross-concept steerability on additional models
    ("Steerability: Qwen-1.5B", "scripts/run_steerability.py", ["--model", "Qwen/Qwen2.5-1.5B"]),
    ("Steerability: Qwen-7B", "scripts/run_steerability.py", ["--model", "Qwen/Qwen2.5-7B"]),
    ("Steerability: Llama-8B", "scripts/run_steerability.py", ["--model", "meta-llama/Llama-3.1-8B"]),
]

DEMOS = [
    ("Refusal demo (Appendix)", "scripts/run_refusal_demo.py"),
    ("Entity steering: Gemma (Section 4.3)", "scripts/run_entity_steering.py"),
    ("Entity steering: OLMo", "scripts/run_entity_steering.py",
     ["--model", "allenai/OLMo-2-0425-1B-Instruct",
      "--results-dir", "results/entity_steering_olmo"]),
]

FIGURES = [
    ("Generate paper figures", "scripts/generate_figures.py"),
]


def load_checkpoint(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def run_script(name, script, extra_args, checkpoint_path, state):
    # Use name as checkpoint key (unique per task, even for same script with different args)
    if name in state["completed"]:
        print(f"\n  [SKIP] {name} (already completed)")
        return True

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    cmd = [sys.executable, script] + extra_args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        state["failed"].append(name)
        save_checkpoint(checkpoint_path, state)
        return False

    state["completed"].append(name)
    state["failed"] = [s for s in state["failed"] if s != name]
    save_checkpoint(checkpoint_path, state)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all LAP experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-replication", action="store_true",
                       help="Skip cross-architecture replication (saves time)")
    parser.add_argument("--skip-demos", action="store_true",
                       help="Skip refusal and entity steering demos")
    parser.add_argument("--reset", action="store_true",
                       help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    checkpoint_path = Path(args.results_dir) / ".run_all_checkpoint.json"

    if args.reset and checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleared.")

    state = load_checkpoint(checkpoint_path)

    if state["completed"]:
        print(f"Resuming: {len(state['completed'])} script(s) already completed.")

    # Scripts have inconsistent arg names, so we build per-script args.
    # --device is accepted by all. --results-dir by all except validate.py
    # and generate_figures.py. Batch size args vary so we skip them
    # and let each script use its own default.

    all_tasks = list(EXPERIMENTS)
    if not args.skip_replication:
        all_tasks.extend(REPLICATION)
    if not args.skip_demos:
        all_tasks.extend(DEMOS)
    all_tasks.extend(FIGURES)

    # Scripts that don't accept --results-dir
    no_results_dir = {"scripts/validate.py", "scripts/generate_figures.py"}
    no_device = {"scripts/run_failure_modes.py", "scripts/run_perturbation.py",
                 "scripts/generate_figures.py"}

    run_failed = []
    for task in all_tasks:
        if len(task) == 3:
            name, script, model_args = task
        else:
            name, script = task
            model_args = []

        script_args = list(model_args)
        if script not in no_device and "--device" not in script_args:
            script_args += ["--device", args.device]
        if script not in no_results_dir and "--results-dir" not in script_args:
            script_args += ["--results-dir", args.results_dir]
        if not run_script(name, script, script_args, checkpoint_path, state):
            run_failed.append(name)

    print(f"\n{'='*60}")
    total = len(all_tasks)
    n_completed = len(state['completed'])
    print(f"  {len(state['completed'])}/{total} completed")
    if run_failed:
        print(f"  Failed this run: {', '.join(run_failed)}")
    else:
        print("  All experiments completed successfully.")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
