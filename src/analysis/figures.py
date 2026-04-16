"""Figure generation for all 6-7 paper figures.

Each figure method produces a self-contained matplotlib figure
matching the specifications in the operational prep document.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

from src.metrics.lap import LAPVector

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FAMILY_COLORS = {
    "factual_recall": "#1f77b4",
    "arithmetic": "#ff7f0e",
    "syntax": "#2ca02c",
    "semantics": "#d62728",
    "safety": "#9467bd",
    "sentiment": "#9467bd",
}

METRIC_LABELS = {
    "a_lin": r"$A_{\mathrm{lin}}$",
    "a_nonlin": r"$A_{\mathrm{nonlin}}$",
    "probe_gap": r"$\Delta$ (Probe Gap)",
    "effective_rank": r"$\rho$ (Effective Rank)",
    "pca_concentration_32": r"$\kappa_{32}$",
    "pca_concentration_128": r"$\kappa_{128}$",
    "condition_number": r"$\mathrm{cond}$",
    "r2_sparse": r"$R^2_{\mathrm{sparse}}$",
    "direction_stability": r"$\delta$ (Dir. Stability)",
    "perturbation_sensitivity": r"$\lambda$ (Pert. Sensitivity)",
}


class FigureGenerator:
    """Generate all paper figures."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def figure1_lap_across_depth(
        self,
        lap_by_family_layer: dict[str, dict[int, list]],
        metrics_to_plot: list[str] | None = None,
        validated_mask: dict[str, dict[int, bool]] | None = None,
    ) -> plt.Figure:
        """Figure 1: LAP metrics across depth, one panel per metric.

        Args:
            lap_by_family_layer: family -> layer -> list of LAPVector
            metrics_to_plot: Which metrics to plot (default: key ones)
            validated_mask: family -> layer -> is_validated
        """
        if metrics_to_plot is None:
            metrics_to_plot = [
                "a_lin", "probe_gap", "direction_stability",
                "condition_number", "effective_rank", "perturbation_sensitivity",
            ]

        n_metrics = len(metrics_to_plot)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), squeeze=False)

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // n_cols][idx % n_cols]

            for family, layer_data in lap_by_family_layer.items():
                layers = sorted(layer_data.keys())
                means = []
                ses = []

                for l in layers:
                    vectors = layer_data[l]
                    values = [getattr(v, metric) for v in vectors]
                    values = [v for v in values if not np.isnan(v)]
                    if values:
                        means.append(np.mean(values))
                        ses.append(np.std(values) / np.sqrt(len(values)))
                    else:
                        means.append(np.nan)
                        ses.append(0)

                means = np.array(means)
                ses = np.array(ses)
                color = FAMILY_COLORS.get(family, "#333333")

                ax.plot(layers, means, color=color, label=family, linewidth=1.5)
                ax.fill_between(
                    layers, means - ses, means + ses, color=color, alpha=0.15
                )

            ax.set_xlabel("Layer")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_title(METRIC_LABELS.get(metric, metric))

        # Legend on first panel
        axes[0][0].legend(loc="best")

        # Hide empty panels
        for idx in range(n_metrics, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle("Figure 1: LAP Across Depth", fontsize=13, y=1.02)
        fig.tight_layout()

        fig.savefig(self.output_dir / "figure1_lap_depth.pdf")
        fig.savefig(self.output_dir / "figure1_lap_depth.png")
        return fig

    def figure2_lap_predicts_across_layers(
        self,
        layers: list[int],
        observed_by_tool: dict[str, np.ndarray],
        predicted_by_tool: dict[str, np.ndarray],
    ) -> plt.Figure:
        """Figure 2: LAP predicts tool success across layers.

        Args:
            layers: Layer indices.
            observed_by_tool: tool_name -> per-layer mean tool success.
            predicted_by_tool: tool_name -> per-layer ridge-predicted success.
        """
        tools = sorted(observed_by_tool.keys())
        n_tools = len(tools)

        fig, axes = plt.subplots(1, n_tools, figsize=(5 * n_tools, 4))
        if n_tools == 1:
            axes = [axes]

        for ax, tool in zip(axes, tools):
            obs = observed_by_tool[tool]
            pred = predicted_by_tool[tool]

            ax.scatter(layers, obs, color="#1f77b4", s=30, zorder=3, label="Observed")
            ax.plot(layers, pred, color="#ff7f0e", linewidth=2, label="LAP-predicted")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Tool Success")
            ax.set_title(tool)
            ax.legend()

        fig.suptitle("Figure 2: LAP Predicts Tool Success Across Layers", fontsize=13, y=1.02)
        fig.tight_layout()

        fig.savefig(self.output_dir / "figure2_lap_predicts_layers.pdf")
        fig.savefig(self.output_dir / "figure2_lap_predicts_layers.png")
        return fig

    def figure3_lap_predicts_across_prompts(
        self,
        lap_index: np.ndarray,
        tool_success: dict[str, np.ndarray],
        families: np.ndarray,
        rho_values: dict[str, float],
        p_values: dict[str, float],
    ) -> plt.Figure:
        """Figure 3: THE KEY FIGURE. LAP index vs tool success per prompt.

        Args:
            lap_index: LAP index (PC1) per prompt.
            tool_success: tool_name -> per-prompt tool success.
            families: Family label per prompt.
            rho_values: Spearman rho per tool.
            p_values: p-values per tool.
        """
        tools = sorted(tool_success.keys())
        n_tools = len(tools)

        fig, axes = plt.subplots(1, n_tools, figsize=(5 * n_tools, 4.5))
        if n_tools == 1:
            axes = [axes]

        unique_families = sorted(set(families))

        for ax, tool in zip(axes, tools):
            y = tool_success[tool]

            for family in unique_families:
                mask = families == family
                color = FAMILY_COLORS.get(family, "#333333")
                ax.scatter(
                    lap_index[mask], y[mask],
                    color=color, label=family, alpha=0.6, s=20,
                )

            ax.set_xlabel("LAP Index (PC1)")
            ax.set_ylabel("Tool Success")
            ax.set_title(tool)

            rho = rho_values.get(tool, 0.0)
            p = p_values.get(tool, 1.0)
            ax.annotate(
                f"Spearman ρ = {rho:.3f}\np = {p:.1e}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.legend(loc="lower right", fontsize=7)

        fig.suptitle(
            "Figure 3: LAP Predicts Tool Success Across Prompts",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()

        fig.savefig(self.output_dir / "figure3_lap_predicts_prompts.pdf")
        fig.savefig(self.output_dir / "figure3_lap_predicts_prompts.png")
        return fig

    def figure4_baseline_comparison(
        self,
        comparisons: dict[str, dict[str, float]],
        metric: str = "r2_cv",
        errors: dict[str, dict[str, float]] | None = None,
    ) -> plt.Figure:
        """Figure 4: Grouped bar chart comparing LAP vs baselines.

        Args:
            comparisons: tool_name -> {predictor_name: metric_value}
            metric: Which metric to plot.
            errors: tool_name -> {predictor_name: SE}
        """
        tools = sorted(comparisons.keys())
        predictors = ["full_lap", "probe_only", "geometry_only", "decomposition_only", "random"]
        predictor_labels = ["Full LAP", "Probe Only", "Geometry Only", "Decomposition Only", "Random"]

        x = np.arange(len(tools))
        width = 0.15
        offsets = np.arange(len(predictors)) - len(predictors) / 2 + 0.5

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = sns.color_palette("Set2", len(predictors))

        for i, (pred, label) in enumerate(zip(predictors, predictor_labels)):
            vals = [comparisons[t].get(pred, 0.0) for t in tools]
            errs = None
            if errors:
                errs = [errors[t].get(pred, 0.0) for t in tools]
            ax.bar(
                x + offsets[i] * width, vals, width,
                label=label, color=colors[i],
                yerr=errs, capsize=3,
            )

        ax.set_xlabel("Tool")
        ax.set_ylabel(f"Held-out {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(tools)
        ax.legend()
        ax.set_title("Figure 4: Baseline Comparison")

        fig.tight_layout()
        fig.savefig(self.output_dir / "figure4_baseline_comparison.pdf")
        fig.savefig(self.output_dir / "figure4_baseline_comparison.png")
        return fig

    def figure5_failure_clusters(
        self,
        clustering_result,
    ) -> plt.Figure:
        """Figure 5: PCA scatter of failed prompts, colored by cluster.

        Args:
            clustering_result: ClusteringResult from FailureClustering.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        coords = clustering_result.pca_coords
        labels = clustering_result.cluster_labels
        clusters = clustering_result.clusters

        colors = sns.color_palette("Set1", clustering_result.n_clusters)

        for c in range(clustering_result.n_clusters):
            mask = labels == c
            cluster_info = clusters[c] if c < len(clusters) else None
            label = f"Cluster {c}"
            if cluster_info:
                label = f"{cluster_info.taxonomy_match} (n={cluster_info.n_prompts})"

            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                color=colors[c], label=label, alpha=0.6, s=30,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Figure 5: Failure Mode Clusters (k={clustering_result.n_clusters}, "
                      f"silhouette={clustering_result.silhouette_score:.2f})")
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(self.output_dir / "figure5_failure_clusters.pdf")
        fig.savefig(self.output_dir / "figure5_failure_clusters.png")
        return fig

    def figure6_chaotic_regime(
        self,
        lambda_values: np.ndarray,
        steering_effectiveness: np.ndarray,
        lap_quintiles: np.ndarray,
    ) -> plt.Figure:
        """Figure 6: λ vs steering effectiveness, colored by LAP quintile.

        Args:
            lambda_values: Perturbation sensitivity per (prompt, layer).
            steering_effectiveness: Steering effect size per (prompt, layer).
            lap_quintiles: LAP quintile (1-5) per (prompt, layer).
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        cmap = plt.cm.RdYlBu_r
        scatter = ax.scatter(
            lambda_values, steering_effectiveness,
            c=lap_quintiles, cmap=cmap, alpha=0.6, s=20,
            vmin=1, vmax=5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("LAP Quintile (1=low, 5=high)")

        ax.set_xlabel(r"Perturbation Sensitivity $\lambda$")
        ax.set_ylabel("Steering Effectiveness")
        ax.set_title("Figure 6: Chaotic Regime Identification")

        fig.tight_layout()
        fig.savefig(self.output_dir / "figure6_chaotic_regime.pdf")
        fig.savefig(self.output_dir / "figure6_chaotic_regime.png")
        return fig

    def figure7_post_training(
        self,
        layers: list[int],
        delta_lap_by_family: dict[str, dict[int, float]],
    ) -> plt.Figure:
        """Figure 7 (optional): ΔLAP (instruct - base) across layers.

        Args:
            layers: Layer indices.
            delta_lap_by_family: family -> layer -> ΔLAP index.
        """
        fig, ax = plt.subplots(figsize=(8, 4))

        for family, layer_deltas in delta_lap_by_family.items():
            lays = sorted(layer_deltas.keys())
            deltas = [layer_deltas[l] for l in lays]
            color = FAMILY_COLORS.get(family, "#333333")
            ax.plot(lays, deltas, color=color, label=family, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\Delta$LAP (Instruct $-$ Base)")
        ax.set_title("Figure 7: Post-Training Effects on LAP")
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.output_dir / "figure7_post_training.pdf")
        fig.savefig(self.output_dir / "figure7_post_training.png")
        return fig
