"""Experiment 4: Chaotic Regime / Perturbation Sensitivity

Formal analysis of the relationship between λ and tool success.
- Scatter plot: λ vs steering effectiveness, colored by LAP quintile
- Quintile analysis: bin prompts by λ, show tool success per quintile
- Test whether high-λ layers coincide with low LAP and steering failure

Most data is already computed in Exp1 and Exp2. This script
assembles and analyzes it.

Usage:
    python scripts/run_experiment4.py --results-dir results
"""

import scripts._env  # noqa: F401 — must be first

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def run_experiment4(args):
    results_dir = Path(args.results_dir) / "exp4"
    results_dir.mkdir(parents=True, exist_ok=True)
    exp1_dir = Path(args.results_dir) / "exp1"
    exp2_dir = Path(args.results_dir) / "exp2"

    # Load all data
    with open(exp1_dir / "layer_accuracy.json") as f:
        exp1_lin = json.load(f)
    with open(exp1_dir / "geometric_metrics.json") as f:
        exp1_geo = json.load(f)
    with open(exp1_dir / "mlp_probe_results.json") as f:
        exp1_mlp = json.load(f)
    with open(exp2_dir / "steering_results.json") as f:
        exp2_steer = json.load(f)

    families = sorted(exp1_lin.keys() & exp2_steer.keys() & exp1_geo.keys())

    # ============================================================
    # Collect per-layer data across all families
    # ============================================================
    all_lambdas = []
    all_lin_accs = []
    all_steering = []
    all_probe_gaps = []
    all_families = []
    all_layers = []

    for family in families:
        lin_layers = exp1_lin[family]["layers"]
        geo_layers = exp1_geo[family]
        steer_layers = exp2_steer[family]["layers"]
        mlp_layers = exp1_mlp.get(family, {})

        common = sorted(set(lin_layers.keys()) & set(geo_layers.keys()) &
                        set(steer_layers.keys()), key=int)

        for l in common:
            lam = geo_layers[l].get("mean_lambda", np.nan)
            la = lin_layers[l]["linear_acc"]
            dp = steer_layers[l]["mean_delta_p"]
            mlp_acc = mlp_layers.get(l, {}).get("mlp_acc", 0)
            gap = mlp_acc - la

            if not np.isnan(lam):
                all_lambdas.append(lam)
                all_lin_accs.append(la)
                all_steering.append(dp)
                all_probe_gaps.append(gap)
                all_families.append(family)
                all_layers.append(int(l))

    all_lambdas = np.array(all_lambdas)
    all_lin_accs = np.array(all_lin_accs)
    all_steering = np.array(all_steering)
    all_probe_gaps = np.array(all_probe_gaps)
    all_families = np.array(all_families)
    all_layers = np.array(all_layers)

    n_points = len(all_lambdas)
    print(f"Total data points: {n_points} (layers × families)")

    # ============================================================
    # LAP quintiles based on linear accuracy
    # ============================================================
    lap_quintiles = np.digitize(
        all_lin_accs,
        bins=np.percentile(all_lin_accs[all_lin_accs > 0],
                          [20, 40, 60, 80]) if (all_lin_accs > 0).sum() > 4
        else [0.01, 0.1, 0.3, 0.5]
    ) + 1
    # Quintile 1 = lowest LAP, 5 = highest

    # ============================================================
    # Overall correlations
    # ============================================================
    print("\n=== OVERALL CORRELATIONS (all families pooled) ===\n")

    rho_ls, p_ls = spearmanr(all_lambdas, all_steering)
    rho_ll, p_ll = spearmanr(all_lambdas, all_lin_accs)
    rho_lg, p_lg = spearmanr(all_lambdas, all_probe_gaps)

    print(f"  λ vs steering ΔP:     ρ = {rho_ls:+.3f} (p = {p_ls:.1e})")
    print(f"  λ vs linear accuracy: ρ = {rho_ll:+.3f} (p = {p_ll:.1e})")
    print(f"  λ vs probe gap:       ρ = {rho_lg:+.3f} (p = {p_lg:.1e})")

    # ============================================================
    # Quintile analysis
    # ============================================================
    print("\n=== QUINTILE ANALYSIS ===\n")
    print(f"{'Quintile':>8} {'n':>4} {'Mean λ':>10} {'Mean LinAcc':>12} "
          f"{'Mean ΔP':>10} {'Mean Gap':>10}")
    print("-" * 60)

    for q in range(1, 6):
        mask = lap_quintiles == q
        if mask.sum() == 0:
            continue
        print(f"{q:>8} {mask.sum():>4} {all_lambdas[mask].mean():>10.1f} "
              f"{all_lin_accs[mask].mean():>12.3f} "
              f"{all_steering[mask].mean():>+10.4f} "
              f"{all_probe_gaps[mask].mean():>+10.3f}")

    # ============================================================
    # Per-family correlations
    # ============================================================
    print("\n=== PER-FAMILY: λ vs Steering ===\n")
    print(f"{'Family':<16} {'ρ(λ,ΔP)':>10} {'p':>10} {'ρ(λ,lin)':>10} {'p':>10}")
    print("-" * 60)

    for family in families:
        mask = all_families == family
        rho1, p1 = spearmanr(all_lambdas[mask], all_steering[mask])
        rho2, p2 = spearmanr(all_lambdas[mask], all_lin_accs[mask])
        print(f"{family:<16} {rho1:>+10.3f} {p1:>10.4f} {rho2:>+10.3f} {p2:>10.4f}")

    # ============================================================
    # L25 anomaly analysis
    # ============================================================
    print("\n=== L25 ANOMALY ===\n")
    print(f"{'Family':<16} {'λ_L24':>8} {'λ_L25':>8} {'Ratio':>7} "
          f"{'Lin_L24':>8} {'Lin_L25':>8} {'ΔP_L24':>8} {'ΔP_L25':>8}")
    print("-" * 80)

    for family in families:
        geo = exp1_geo[family]
        lin_l = exp1_lin[family]["layers"]
        steer_l = exp2_steer[family]["layers"]

        l24 = geo.get("24", {}).get("mean_lambda", np.nan)
        l25 = geo.get("25", {}).get("mean_lambda", np.nan)
        ratio = l25 / l24 if l24 > 0 else np.nan

        lin24 = lin_l.get("24", {}).get("linear_acc", 0)
        lin25 = lin_l.get("25", {}).get("linear_acc", 0)
        dp24 = steer_l.get("24", {}).get("mean_delta_p", 0)
        dp25 = steer_l.get("25", {}).get("mean_delta_p", 0)

        print(f"{family:<16} {l24:>8.0f} {l25:>8.0f} {ratio:>6.1f}x "
              f"{lin24:>8.3f} {lin25:>8.3f} {dp24:>+8.4f} {dp25:>+8.4f}")

    # ============================================================
    # Figures
    # ============================================================
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 6: λ vs steering effectiveness, colored by LAP quintile
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.RdYlBu_r
    scatter = ax.scatter(
        all_lambdas, all_steering,
        c=lap_quintiles, cmap=cmap, alpha=0.7, s=40,
        vmin=1, vmax=5, edgecolors="gray", linewidths=0.3,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("LAP Quintile (1=low, 5=high)")

    ax.set_xlabel(r"Perturbation Sensitivity $\lambda$", fontsize=12)
    ax.set_ylabel("Steering Effect (ΔP)", fontsize=12)
    ax.set_title("Figure 6: Chaotic Regime Identification", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "figure6_chaotic_regime.png", dpi=150)
    fig.savefig(fig_dir / "figure6_chaotic_regime.pdf")
    plt.close(fig)
    print(f"\nSaved Figure 6 to {fig_dir}")

    # Figure: λ trajectory across layers (all families)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = sns.color_palette("Set2", len(families))
    for i, family in enumerate(families):
        geo = exp1_geo[family]
        layers_sorted = sorted(geo.keys(), key=int)
        lambdas = [geo[l]["mean_lambda"] for l in layers_sorted]
        layer_nums = [int(l) for l in layers_sorted]

        axes[0].plot(layer_nums, lambdas, color=colors[i], label=family, linewidth=1.5)

        lin_l = exp1_lin[family]["layers"]
        lin_accs = [lin_l.get(l, {}).get("linear_acc", 0) for l in layers_sorted]
        axes[1].plot(layer_nums, lin_accs, color=colors[i], label=family, linewidth=1.5)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel(r"$\lambda$ (perturbation sensitivity)")
    axes[0].set_title(r"$\lambda$ decreases with depth (stabilization)")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Linear Accuracy (unembedding)")
    axes[1].set_title("Linear accessibility emerges at late layers")
    axes[1].legend(fontsize=8)

    fig.suptitle("Experiment 4: λ and Linear Accessibility Across Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_lambda_trajectory.png", dpi=150)
    fig.savefig(fig_dir / "figure_lambda_trajectory.pdf")
    plt.close(fig)

    # Save results
    results_summary = {
        "overall_correlations": {
            "lambda_vs_steering": {"rho": float(rho_ls), "p": float(p_ls)},
            "lambda_vs_linear_acc": {"rho": float(rho_ll), "p": float(p_ll)},
            "lambda_vs_probe_gap": {"rho": float(rho_lg), "p": float(p_lg)},
        },
        "n_datapoints": n_points,
        "families": families,
    }

    with open(results_dir / "chaotic_regime_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nExperiment 4 complete. Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Chaotic Regime")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_experiment4(args)


if __name__ == "__main__":
    main()
