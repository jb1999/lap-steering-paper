"""LAP metric definitions and computation.

Implements all 10 LAP features per (prompt, layer):
  1. A_lin          - Linear probe accuracy
  2. A_nonlin       - MLP probe accuracy
  3. Δ              - Probe gap (A_nonlin - A_lin)
  4. ρ              - Effective rank (RANKME)
  5. κ_32           - PCA concentration (top-32)
  6. κ_128          - PCA concentration (top-128)
  7. cond           - Condition number of concept-relevant subspace
  8. R²_sparse      - Transcoder reconstruction R²
  9. δ              - Direction stability (split-half cosine)
  10. λ             - Perturbation sensitivity
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass
class LAPVector:
    """The 10-feature LAP vector for a single (prompt, layer) pair."""
    a_lin: float
    a_nonlin: float
    probe_gap: float  # Δ
    effective_rank: float  # ρ
    pca_concentration_32: float  # κ_32
    pca_concentration_128: float  # κ_128
    condition_number: float  # cond
    r2_sparse: float  # R²_sparse
    direction_stability: float  # δ
    perturbation_sensitivity: float  # λ

    def to_array(self) -> np.ndarray:
        return np.array([
            self.a_lin,
            self.a_nonlin,
            self.probe_gap,
            self.effective_rank,
            self.pca_concentration_32,
            self.pca_concentration_128,
            self.condition_number,
            self.r2_sparse,
            self.direction_stability,
            self.perturbation_sensitivity,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "a_lin", "a_nonlin", "probe_gap",
            "effective_rank", "pca_concentration_32", "pca_concentration_128",
            "condition_number", "r2_sparse",
            "direction_stability", "perturbation_sensitivity",
        ]


class LAPMetrics:
    """Compute individual LAP metrics from activation matrices."""

    @staticmethod
    def effective_rank(H: np.ndarray) -> float:
        """RANKME: exp(H(σ̃)) where σ̃ are normalized singular values.

        Args:
            H: Activation matrix of shape (n_samples, d_model).
        """
        # Center the data
        H_centered = H - H.mean(axis=0)
        sv = np.linalg.svd(H_centered, compute_uv=False)
        # Normalize to probability distribution
        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) == 0:
            return 1.0
        sv_norm = sv_pos / sv_pos.sum()
        return float(np.exp(entropy(sv_norm)))

    @staticmethod
    def pca_concentration(H: np.ndarray, k: int) -> float:
        """Fraction of variance captured by top-k PCA components.

        Args:
            H: Activation matrix of shape (n_samples, d_model).
            k: Number of top components.
        """
        H_centered = H - H.mean(axis=0)
        sv = np.linalg.svd(H_centered, compute_uv=False)
        sv_sq = sv ** 2
        total_var = sv_sq.sum()
        if total_var < 1e-10:
            return 1.0
        top_k = min(k, len(sv_sq))
        return float(sv_sq[:top_k].sum() / total_var)

    @staticmethod
    def condition_number_concept_subspace(
        H: np.ndarray,
        labels: np.ndarray,
        n_pcs: int = 15,
    ) -> float:
        """Condition number of the concept-relevant subspace.

        Per operational definition in Section 3.2:
        1. Compute difference-of-means direction d_c
        2. Project out d_c from all activations
        3. Take top-15 PCs of the residual
        4. Form 16-dim subspace, compute condition number

        Args:
            H: Activation matrix (n_samples, d_model).
            labels: Binary concept labels (n_samples,).
            n_pcs: Number of residual PCs (default 15).
        """
        # Difference of means direction
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return float("inf")

        d_c = H[pos_mask].mean(axis=0) - H[neg_mask].mean(axis=0)
        d_c_norm = np.linalg.norm(d_c)
        if d_c_norm < 1e-10:
            return float("inf")
        d_c = d_c / d_c_norm

        # Project out d_c
        projections = H @ d_c  # (n_samples,)
        H_residual = H - np.outer(projections, d_c)

        # Top PCs of residual
        H_res_centered = H_residual - H_residual.mean(axis=0)
        U, S, Vt = np.linalg.svd(H_res_centered, full_matrices=False)

        # Form subspace: d_c + top n_pcs PCs
        n_pcs_actual = min(n_pcs, len(S))
        subspace_dirs = np.vstack([d_c.reshape(1, -1), Vt[:n_pcs_actual]])  # (1+n_pcs, d)

        # Project activations onto subspace
        P = H @ subspace_dirs.T  # (n_samples, 1+n_pcs)

        # Condition number
        sv_P = np.linalg.svd(P, compute_uv=False)
        if sv_P[-1] < 1e-10:
            return float("inf")
        return float(sv_P[0] / sv_P[-1])

    @staticmethod
    def reconstruction_r2(
        H: np.ndarray,
        H_reconstructed: np.ndarray,
    ) -> np.ndarray:
        """Per-prompt sparse reconstruction R².

        Args:
            H: Original activations (n_samples, d_model).
            H_reconstructed: Reconstructed activations (n_samples, d_model).

        Returns:
            R² values per prompt (n_samples,).
        """
        H_mean = H.mean(axis=0)
        ss_res = np.sum((H - H_reconstructed) ** 2, axis=1)
        ss_tot = np.sum((H - H_mean) ** 2, axis=1)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        return r2

    @staticmethod
    def perturbation_sensitivity(
        extractor,
        prompts: list[str],
        layer: int,
        n_perturbations: int = 10,
        alpha_scale: float = 0.01,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Per-prompt perturbation sensitivity λ.

        Monte Carlo estimate of local amplification rate.

        Args:
            extractor: ActivationExtractor instance.
            prompts: List of prompts.
            layer: Layer at which to perturb.
            n_perturbations: Number of random perturbation directions.
            alpha_scale: Scale factor (α = alpha_scale * ||h_l||).

        Returns:
            λ values per prompt (n_prompts,).
        """
        import torch

        # Get baseline activations and output
        result = extractor.extract(prompts, layers=[layer], batch_size=batch_size)
        h_l = result.activations[layer]  # (n_prompts, d_model)
        baseline_output = extractor.get_baseline_output(prompts, batch_size=batch_size)

        d_model = h_l.shape[1]
        n_prompts = len(prompts)

        amplifications = np.zeros((n_prompts, n_perturbations))

        rng = np.random.RandomState(42)

        for p in range(n_perturbations):
            # Random unit perturbation
            eps = rng.randn(n_prompts, d_model).astype(np.float32)
            eps = eps / np.linalg.norm(eps, axis=1, keepdims=True)

            # Scale by alpha * ||h_l||
            h_norms = torch.norm(h_l.float(), dim=1).numpy()
            alpha = alpha_scale * h_norms  # (n_prompts,)
            perturbation = torch.tensor(eps * alpha[:, None])

            # Get perturbed output
            perturbed_output = extractor.extract_with_perturbation(
                prompts, layer, perturbation, batch_size=batch_size
            )

            # Compute amplification
            delta_output = (perturbed_output.float() - baseline_output.float()).numpy()
            output_norms = np.linalg.norm(delta_output, axis=1)
            amplifications[:, p] = output_norms / (alpha + 1e-10)

        return amplifications.mean(axis=1)
"""LAP computation pipeline.

Given activations + probes + (optional) transcoder reconstructions,
computes the full 10-feature LAP vector per (prompt, layer).

Parallelized across layers using spawn workers to avoid BLAS deadlocks.
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.metrics.lap import LAPMetrics, LAPVector
from src.probes.trainer import ProbeResult


def _compute_lap_for_layer(
    H: np.ndarray,
    labels: np.ndarray,
    a_lin: float,
    a_nonlin: float,
    probe_gap: float,
    direction_stability: float,
    perturbation_sensitivity: np.ndarray | None,
    layer_idx: int,
) -> dict:
    """Compute LAP metrics for one layer. Runs in a spawn worker."""
    n_samples = H.shape[0]

    eff_rank = LAPMetrics.effective_rank(H)
    kappa_32 = LAPMetrics.pca_concentration(H, k=32)
    kappa_128 = LAPMetrics.pca_concentration(H, k=128)
    cond = LAPMetrics.condition_number_concept_subspace(H, labels)

    lambda_values = perturbation_sensitivity if perturbation_sensitivity is not None else np.full(n_samples, np.nan)

    return {
        "layer_idx": layer_idx,
        "eff_rank": eff_rank,
        "kappa_32": kappa_32,
        "kappa_128": kappa_128,
        "cond": cond,
        "a_lin": a_lin,
        "a_nonlin": a_nonlin,
        "probe_gap": probe_gap,
        "direction_stability": direction_stability,
        "lambda_values": lambda_values,
    }


def compute_lap_all_layers(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    probe_results: dict[int, ProbeResult],
    reconstructions: dict[int, np.ndarray] | None = None,
    perturbation_sensitivities: dict[int, np.ndarray] | None = None,
    n_workers: int | None = None,
) -> dict[int, list[LAPVector]]:
    """Compute LAP vectors across all layers, parallelized.

    Args:
        activations: Dict mapping layer -> (n_samples, d_model).
        labels: Binary concept labels (n_samples,).
        probe_results: Dict mapping layer -> ProbeResult.
        reconstructions: Dict mapping layer -> (n_samples, d_model) reconstructions.
        perturbation_sensitivities: Dict mapping layer -> (n_samples,) λ values.
        n_workers: Number of parallel workers.

    Returns:
        Dict mapping layer -> list of LAPVector.
    """
    layers = sorted(activations.keys())
    n_layers = len(layers)

    if n_workers is None:
        n_workers = min(n_layers, max(1, mp.cpu_count() - 2))

    # Submit all layers to spawn workers
    ctx = mp.get_context("spawn")
    layer_results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {}
        for layer_idx in layers:
            pr = probe_results[layer_idx]
            lambda_vals = None
            if perturbation_sensitivities is not None and layer_idx in perturbation_sensitivities:
                lambda_vals = perturbation_sensitivities[layer_idx]

            f = executor.submit(
                _compute_lap_for_layer,
                activations[layer_idx], labels,
                pr.a_lin, pr.a_nonlin, pr.probe_gap, pr.direction_stability,
                lambda_vals, layer_idx,
            )
            futures[f] = layer_idx

        for f in as_completed(futures):
            layer_idx = futures[f]
            layer_results[layer_idx] = f.result()

    # Assemble LAPVectors
    result = {}
    for layer_idx in layers:
        lr = layer_results[layer_idx]
        n_samples = activations[layer_idx].shape[0]

        # Per-prompt R²
        if reconstructions is not None and layer_idx in reconstructions:
            r2_values = LAPMetrics.reconstruction_r2(
                activations[layer_idx], reconstructions[layer_idx]
            )
        else:
            r2_values = np.full(n_samples, np.nan)

        vectors = []
        for i in range(n_samples):
            vectors.append(LAPVector(
                a_lin=lr["a_lin"],
                a_nonlin=lr["a_nonlin"],
                probe_gap=lr["probe_gap"],
                effective_rank=lr["eff_rank"],
                pca_concentration_32=lr["kappa_32"],
                pca_concentration_128=lr["kappa_128"],
                condition_number=lr["cond"],
                r2_sparse=float(r2_values[i]),
                direction_stability=lr["direction_stability"],
                perturbation_sensitivity=float(lr["lambda_values"][i]),
            ))

        result[layer_idx] = vectors

    return result


def lap_to_matrix(
    lap_vectors: dict[int, list[LAPVector]],
) -> tuple[np.ndarray, list[int], list[str]]:
    """Convert LAP vectors to a feature matrix.

    Returns:
        X: Feature matrix (n_layers * n_prompts, 10).
        layer_indices: Layer index for each row.
        feature_names: Names of the 10 features.
    """
    feature_names = LAPVector.feature_names()
    rows = []
    layer_indices = []

    for layer_idx in sorted(lap_vectors.keys()):
        for vec in lap_vectors[layer_idx]:
            rows.append(vec.to_array())
            layer_indices.append(layer_idx)

    return np.array(rows), layer_indices, feature_names
