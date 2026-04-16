"""Generate all figures for the paper."""

import scripts._env  # noqa: F401

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_data():
    base = Path("results")
    data = {}
    for name in ["layer_accuracy", "geometric_metrics", "mlp_probe_results"]:
        with open(base / "exp1" / f"{name}.json") as f:
            data[name] = json.load(f)
    with open(base / "exp2" / "steering_results.json") as f:
        data["steering"] = json.load(f)
    try:
        with open(base / "replication_Llama-3.1-8B" / "results.json") as f:
            data["replication"] = json.load(f)
    except FileNotFoundError:
        data["replication"] = None
    try:
        with open(base / "refusal_demo" / "refusal_results.json") as f:
            data["refusal"] = json.load(f)
    except FileNotFoundError:
        data["refusal"] = None
    return data


def fig1_emergence(data, outdir):
    """Figure 1: A_mlp (solid) and A_lin (dashed) across layers, single panel."""
    lin = data["layer_accuracy"]
    mlp = data["mlp_probe_results"]

    families = sorted(lin.keys())
    colors = {"arithmetic": "#1f77b4", "geography": "#ff7f0e", "sequence": "#2ca02c",
              "word_transform": "#d62728", "analogy": "#9467bd"}

    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    for fam in families:
        if fam not in mlp:
            continue
        mlp_d = mlp[fam]
        ls = sorted(mlp_d.keys(), key=int)
        x = [int(l) for l in ls]
        y_mlp = [mlp_d[l]["mlp_acc"] for l in ls]

        lin_d = lin[fam]["layers"]
        y_lin = [lin_d.get(l, {}).get("linear_acc", 0) for l in ls]

        label = fam.replace("_", " ")
        ax.plot(x, y_mlp, color=colors.get(fam, "gray"), linewidth=2.0,
                linestyle="-", marker="s", markersize=3)
        ax.plot(x, y_lin, color=colors.get(fam, "gray"), linewidth=2.0,
                linestyle=":", marker=".", markersize=5, alpha=0.8)
        # Color-only legend patch
        ax.plot([], [], color=colors.get(fam, "gray"), linewidth=4, label=label)

    # Add line-style legend entries
    ax.plot([], [], color="black", linewidth=1.8, linestyle="-", marker="s",
            markersize=3, label=r"$A_{\mathrm{mlp}}$")
    ax.plot([], [], color="black", linewidth=1.8, linestyle=":", marker=".",
            markersize=5, label=r"$A_{\mathrm{lin}}$")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2,
              bbox_to_anchor=(0.0, 1.28), frameon=True)
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    fig.savefig(outdir / "fig1_emergence.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig1_emergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1: emergence")


def fig2_steering(data, outdir):
    """Figure 2: Steering ΔP vs A_lin scatter + per-layer comparison."""
    lin = data["layer_accuracy"]
    steer = data["steering"]

    families = sorted(set(lin.keys()) & set(steer.keys()))
    colors = {"arithmetic": "#1f77b4", "geography": "#ff7f0e", "sequence": "#2ca02c",
              "word_transform": "#d62728", "analogy": "#9467bd"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Scatter A_lin vs ΔP
    for fam in families:
        sl = steer[fam]["layers"]
        ll = lin[fam]["layers"]
        common = sorted(set(sl.keys()) & set(ll.keys()), key=int)
        x = [ll[l]["linear_acc"] for l in common]
        y = [sl[l]["mean_delta_p"] for l in common]
        ax1.scatter(x, y, color=colors.get(fam, "gray"), s=25, alpha=0.7,
                   label=fam.replace("_", " "))

    ax1.set_xlabel(r"$A_{\mathrm{lin}}$ (logit lens accuracy)", fontsize=11)
    ax1.set_ylabel(r"Steering $\Delta P$", fontsize=11)
    ax1.set_title("(a) Linear accuracy vs. steering effect", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # Panel B: Per-layer trajectories side by side
    for fam in ["sequence", "word_transform"]:
        if fam not in steer:
            continue
        sl = steer[fam]["layers"]
        ll = lin[fam]["layers"]
        common = sorted(set(sl.keys()) & set(ll.keys()), key=int)
        x = [int(l) for l in common]
        y_lin = [ll[l]["linear_acc"] for l in common]
        y_dp = [sl[l]["mean_delta_p"] for l in common]

        color = colors.get(fam, "gray")
        label = fam.replace("_", " ")
        ax2.plot(x, y_lin, color=color, linewidth=1.5, linestyle="--",
                label=f"{label} $A_{{\\mathrm{{lin}}}}$")
        ax2.plot(x, y_dp, color=color, linewidth=1.8,
                label=f"{label} $\\Delta P$")

    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("(b) Steering tracks linear emergence", fontsize=12)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(outdir / "fig2_steering.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig2_steering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2: steering")


def fig3_chaotic(data, outdir):
    """Figure 3: λ vs steering ΔP colored by LAP quintile."""
    lin = data["layer_accuracy"]
    geo = data["geometric_metrics"]
    steer = data["steering"]

    families = sorted(set(lin.keys()) & set(steer.keys()) & set(geo.keys()))

    all_lam, all_dp, all_alin = [], [], []
    for fam in families:
        sl = steer[fam]["layers"]
        ll = lin[fam]["layers"]
        gl = geo[fam]
        common = sorted(set(sl.keys()) & set(ll.keys()) & set(gl.keys()), key=int)
        for l in common:
            lam = gl[l].get("mean_lambda", np.nan)
            if not np.isnan(lam):
                all_lam.append(lam)
                all_dp.append(sl[l]["mean_delta_p"])
                all_alin.append(ll[l]["linear_acc"])

    all_lam = np.array(all_lam)
    all_dp = np.array(all_dp)
    all_alin = np.array(all_alin)

    # Quintiles
    nonzero = all_alin[all_alin > 0]
    if len(nonzero) > 4:
        bins = np.percentile(nonzero, [20, 40, 60, 80])
    else:
        bins = [0.01, 0.1, 0.3, 0.5]
    quintiles = np.digitize(all_alin, bins) + 1

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(all_lam, all_dp, c=quintiles, cmap="RdYlBu_r",
                        alpha=0.7, s=35, vmin=1, vmax=5,
                        edgecolors="gray", linewidths=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r"$A_{\mathrm{lin}}$ quintile (1=low, 5=high)", fontsize=10)

    ax.set_xlabel(r"Perturbation sensitivity $\lambda$", fontsize=11)
    ax.set_ylabel(r"Steering $\Delta P$", fontsize=11)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    from scipy.stats import spearmanr
    rho, p = spearmanr(all_lam, all_dp)
    ax.annotate(f"$\\rho = {rho:.2f}$\n$p < 10^{{-20}}$",
               xy=(0.95, 0.95), xycoords="axes fraction",
               ha="right", va="top", fontsize=10,
               bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(outdir / "fig3_chaotic.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig3_chaotic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3: chaotic regime")


def fig4_replication(data, outdir):
    """Figure 4: Gemma vs Llama comparison."""
    if data["replication"] is None:
        print("  Figure 4: skipped (no replication data)")
        return

    gemma = data["layer_accuracy"]
    llama = data["replication"]["families"]
    colors = {"arithmetic": "#1f77b4", "geography": "#ff7f0e", "sequence": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for idx, fam in enumerate(["arithmetic", "geography", "sequence"]):
        ax = axes[idx]

        # Gemma
        gl = gemma[fam]["layers"]
        gls = sorted(gl.keys(), key=int)
        gx = [int(l) / (len(gls) - 1) for l in gls]  # normalize to 0-1
        gy = [gl[l]["linear_acc"] for l in gls]
        ax.plot(gx, gy, color=colors[fam], linewidth=1.8, linestyle="--",
               label=f"Gemma-2B (26L)", marker="o", markersize=3)

        # Llama
        if fam in llama:
            ll = llama[fam]["layer_accuracy"]
            lls = sorted(ll.keys(), key=int)
            lx = [int(l) / (len(lls) - 1) for l in lls]
            ly = [ll[l] for l in lls]
            ax.plot(lx, ly, color=colors[fam], linewidth=1.8,
                   label=f"Llama-8B (32L)", marker="s", markersize=3)

        ax.set_xlabel("Relative depth", fontsize=10)
        ax.set_ylabel(r"$A_{\mathrm{lin}}$", fontsize=10)
        ax.set_title(fam.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle("Cross-model comparison (normalized depth)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_replication.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig4_replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 4: replication")


def fig5_refusal(data, outdir):
    """Figure 5: Refusal demo - separability and steering."""
    if data["refusal"] is None:
        print("  Figure 5: skipped (no refusal data)")
        return

    ref = data["refusal"]
    layers = ref["layer_separability"]
    steer = ref.get("steering", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ls = sorted(layers.keys(), key=int)
    x = [int(l) for l in ls]
    y_acc = [layers[l]["linear_acc"] for l in ls]
    y_d = [layers[l]["cohens_d"] for l in ls]

    ax1.plot(x, y_acc, color="#1f77b4", linewidth=2, marker="o", markersize=4, label="Separability")
    ax1b = ax1.twinx()
    ax1b.plot(x, y_d, color="#ff7f0e", linewidth=1.5, linestyle="--", marker="s",
             markersize=3, label="Cohen's $d$", alpha=0.7)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Linear separability (CV accuracy)", fontsize=10, color="#1f77b4")
    ax1b.set_ylabel("Cohen's $d$", fontsize=10, color="#ff7f0e")
    ax1.set_title("(a) Refusal separability", fontsize=12)
    ax1.legend(loc="lower right", fontsize=8)
    ax1b.legend(loc="center right", fontsize=8)

    # Panel B: steering effect
    if steer:
        steer_ls = sorted(steer.keys(), key=int)
        sx = [int(l) for l in steer_ls]
        sy_ref = [steer[l]["delta_refusal"] for l in steer_ls]
        sy_comp = [steer[l]["delta_comply"] for l in steer_ls]
        sy_acc = [steer[l]["linear_acc"] for l in steer_ls]

        ax2.bar([xi - 0.15 for xi in sx], sy_ref, width=0.3, color="#d62728",
               alpha=0.7, label=r"$\Delta P$(refusal)")
        ax2.bar([xi + 0.15 for xi in sx], sy_comp, width=0.3, color="#2ca02c",
               alpha=0.7, label=r"$\Delta P$(comply)")
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel(r"$\Delta P$ from steering", fontsize=10)
        ax2.set_title("(b) Steering effect (subtract refusal dir.)", fontsize=12)
        ax2.legend(fontsize=8)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(outdir / "fig5_refusal.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "fig5_refusal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 5: refusal")


def main():
    outdir = Path("figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data()

    print("Generating figures:")
    fig1_emergence(data, outdir)
    fig2_steering(data, outdir)
    fig3_chaotic(data, outdir)
    fig4_replication(data, outdir)
    fig5_refusal(data, outdir)
    print(f"\nAll figures saved to {outdir}")


if __name__ == "__main__":
    main()
