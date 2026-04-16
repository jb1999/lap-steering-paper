"""Scaling analysis: how does A_lin and steerability change with model size?

Runs the cross-concept analysis on multiple Pythia model sizes to test
whether concepts become more linearly accessible (and more steerable)
in larger models.

Uses Pythia-160M, 410M, 1B, 2.8B (already done), and optionally 6.9B.

Usage:
    python scripts/run_scaling.py --device cuda
    python scripts/run_scaling.py --device cuda --sizes 160m,410m,1b,2.8b
"""

import scripts._env  # noqa: F401

import argparse
import json
import gc
import torch
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# Import the cross-concept analysis function
from run_steerability import main as cross_concept_main


PYTHIA_MODELS = {
    "70m": "EleutherAI/pythia-70m-deduped",
    "160m": "EleutherAI/pythia-160m-deduped",
    "410m": "EleutherAI/pythia-410m-deduped",
    "1b": "EleutherAI/pythia-1b-deduped",
    "2.8b": "EleutherAI/pythia-2.8b-deduped",
    "6.9b": "EleutherAI/pythia-6.9b-deduped",
}


def main():
    parser = argparse.ArgumentParser(description="Scaling analysis across Pythia sizes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sizes", default="160m,410m,1b,2.8b,6.9b",
                        help="Comma-separated Pythia sizes to test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    results_dir = Path(args.results_dir)

    # Run cross-concept for each size
    for size in sizes:
        model_name = PYTHIA_MODELS[size]
        model_tag = model_name.split("/")[-1]
        result_file = results_dir / "cross_concept" / model_tag / "results.json"

        if result_file.exists() and not args.force:
            print(f"\n{size}: results exist, skipping (use --force to rerun)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running Pythia-{size}")
        print(f"{'=' * 60}")

        # Build args for cross-concept script
        import sys
        sys.argv = [
            "run_cross_concept.py",
            "--model", model_name,
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--results-dir", args.results_dir,
        ]
        if args.force:
            sys.argv.append("--force")

        cross_concept_main()

        gc.collect()
        torch.cuda.empty_cache()

    # Analyze scaling
    print(f"\n{'=' * 60}")
    print("SCALING ANALYSIS")
    print(f"{'=' * 60}")

    scaling_data = {}
    for size in sizes:
        model_name = PYTHIA_MODELS[size]
        model_tag = model_name.split("/")[-1]
        result_file = results_dir / "cross_concept" / model_tag / "results.json"

        if not result_file.exists():
            continue

        with open(result_file) as f:
            r = json.load(f)

        controlled = {f: d for f, d in r["families"].items()
                      if f.startswith("c_") and d.get("max_steering_dp") is not None}

        alins = [d["best_alin"] for d in controlled.values()]
        dps = [d["max_steering_dp"] for d in controlled.values()]

        scaling_data[size] = {
            "n_families": len(controlled),
            "mean_alin": float(np.mean(alins)),
            "median_alin": float(np.median(alins)),
            "max_alin": float(np.max(alins)),
            "mean_dp": float(np.mean(dps)),
            "n_steerable": sum(1 for a in alins if a > 0.1),
            "rho_alin_dp": float(spearmanr(alins, dps)[0]) if len(alins) >= 5 else None,
            "families": {f: {"alin": d["best_alin"], "dp": d["max_steering_dp"]}
                         for f, d in controlled.items()},
        }

    # Print scaling table
    print(f"\n{'Size':<8} {'N':>4} {'Mean A_lin':>10} {'Med A_lin':>10} {'Max A_lin':>10} "
          f"{'Mean ΔP':>8} {'#Steer':>7} {'ρ(A,ΔP)':>8}")
    print("-" * 70)
    for size in sizes:
        if size not in scaling_data:
            continue
        d = scaling_data[size]
        rho_str = f"{d['rho_alin_dp']:+.3f}" if d['rho_alin_dp'] is not None else "---"
        print(f"{size:<8} {d['n_families']:>4} {d['mean_alin']:>10.3f} {d['median_alin']:>10.3f} "
              f"{d['max_alin']:>10.3f} {d['mean_dp']:>8.4f} {d['n_steerable']:>7} {rho_str:>8}")

    # Per-family scaling
    if len(scaling_data) >= 3:
        # Get families common to all sizes
        common = None
        for size, d in scaling_data.items():
            fams = set(d["families"].keys())
            common = fams if common is None else common & fams

        if common:
            print(f"\nPer-family A_lin across sizes ({len(common)} common families):")
            print(f"{'Family':<16}", end="")
            for size in sizes:
                if size in scaling_data:
                    print(f" {size:>8}", end="")
            print()
            print("-" * (16 + 9 * len([s for s in sizes if s in scaling_data])))

            for fam in sorted(common):
                print(f"{fam:<16}", end="")
                for size in sizes:
                    if size in scaling_data:
                        a = scaling_data[size]["families"].get(fam, {}).get("alin", 0)
                        print(f" {a:>8.3f}", end="")
                print()

    # Save
    save_file = results_dir / "scaling" / "scaling_results.json"
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\nResults saved to {save_file}")


if __name__ == "__main__":
    main()
