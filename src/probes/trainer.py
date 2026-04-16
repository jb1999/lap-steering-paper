"""Probe training pipeline with full controls.

Implements: 5-fold CV, regularization sweeps, permutation nulls,
sample-size curves, dimensionality controls.

CPU probes (liblinear, PCA-128) are parallelized across layers.
MLP probes run sequentially on GPU.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from src.probes.linear_probe import LinearProbe
from src.probes.mlp_probe import MLPProbe


@dataclass
class ProbeResult:
    """Result of probe training for one concept at one layer."""

    layer: int
    concept: str
    family: str

    # Core metrics (from full-dim liblinear probe)
    a_lin: float  # Linear probe accuracy (mean across folds)
    a_lin_se: float  # Standard error
    a_nonlin: float  # MLP probe accuracy (mean across folds)
    a_nonlin_se: float
    probe_gap: float  # a_nonlin - a_lin

    # PCA-128 probe results (for robustness comparison)
    a_lin_pca: float = 0.5
    a_lin_pca_se: float = 0.0

    # Weight vectors from linear probe (per fold)
    weight_vectors: list[np.ndarray] = field(default_factory=list)

    # Direction stability: min cosine similarity across split-half
    direction_stability: float = 0.0

    # Permutation null
    a_lin_null: float = 0.5  # Accuracy with shuffled labels
    a_lin_null_se: float = 0.0

    # Sample-size curve
    sample_size_curve: dict[float, float] = field(default_factory=dict)

    # Dimensionality control
    dim_control: dict[int, float] = field(default_factory=dict)

    # Best regularization
    best_C: float = 1.0


def _has_both_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) >= 2


def _train_cpu_probes_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    concept: str,
    family: str,
    n_folds: int = 5,
    n_permutations: int = 5,
    sample_fractions: tuple = (0.25, 0.5, 0.75, 1.0),
    pca_dims: tuple = (16, 32, 64, 128),
    random_state: int = 42,
    run_controls: bool = True,
) -> dict:
    """Train CPU probes (liblinear + PCA-128) for one layer. Runs in a worker process."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    lin_scores = []          # PCA-128 (primary, fast)
    weight_vectors = []
    best_Cs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if not _has_both_classes(y_train):
            continue

        # PCA-128 linear probe (primary — fast and robust)
        lp = LinearProbe(pca_dims=128, random_state=random_state)
        lp.fit(X_train, y_train)
        lin_scores.append(lp.score(X_test, y_test))
        weight_vectors.append(lp.weight_vectors)
        best_Cs.append(lp.best_C)

    if not lin_scores:
        lin_scores = [0.5]

    n_completed = len(lin_scores)
    result = {
        "layer": layer,
        "a_lin": float(np.mean(lin_scores)),
        "a_lin_se": float(np.std(lin_scores) / np.sqrt(n_completed)),
        "a_lin_pca": float(np.mean(lin_scores)),  # PCA is primary
        "a_lin_pca_se": float(np.std(lin_scores) / np.sqrt(n_completed)),
        "weight_vectors": weight_vectors,
        "best_C": float(np.mean(best_Cs)),
    }

    # Direction stability
    if len(weight_vectors) >= 2:
        min_cos = 1.0
        for i in range(len(weight_vectors)):
            for j in range(i + 1, len(weight_vectors)):
                w1, w2 = weight_vectors[i], weight_vectors[j]
                # Flatten for multi-class (n_classes, d_model) -> 1D
                w1f, w2f = w1.ravel(), w2.ravel()
                cos = np.dot(w1f, w2f) / (np.linalg.norm(w1f) * np.linalg.norm(w2f) + 1e-10)
                min_cos = min(min_cos, cos)
        result["direction_stability"] = float(min_cos)
    else:
        result["direction_stability"] = 1.0

    # Controls
    if run_controls:
        # Permutation null
        rng = np.random.RandomState(random_state)
        null_scores = []
        split = int(0.8 * len(X))
        for _ in range(n_permutations):
            y_shuffled = rng.permutation(y)
            if not _has_both_classes(y_shuffled[:split]):
                continue
            lp2 = LinearProbe(pca_dims=128, random_state=random_state)
            lp2.fit(X[:split], y_shuffled[:split])
            null_scores.append(lp2.score(X[split:], y_shuffled[split:]))
        result["a_lin_null"] = float(np.mean(null_scores)) if null_scores else 0.5
        result["a_lin_null_se"] = float(np.std(null_scores)) if null_scores else 0.0

        # Sample-size curve
        curve = {}
        X_test_ss, y_test_ss = X[split:], y[split:]
        for frac in sample_fractions:
            n = max(4, int(frac * split))
            if not _has_both_classes(y[:n]):
                curve[frac] = float("nan")
                continue
            lp3 = LinearProbe(pca_dims=128, random_state=random_state)
            lp3.fit(X[:n], y[:n])
            curve[frac] = lp3.score(X_test_ss, y_test_ss)
        result["sample_size_curve"] = curve

        # Dimensionality control
        control = {}
        if _has_both_classes(y[:split]):
            for k in pca_dims:
                if k >= X.shape[1] or k >= split:
                    continue
                pca = PCA(n_components=k, random_state=random_state)
                X_pca = pca.fit_transform(X[:split])
                X_test_pca = pca.transform(X[split:])
                lp4 = LinearProbe(solver="lbfgs", random_state=random_state)
                lp4.fit(X_pca, y[:split])
                control[k] = lp4.score(X_test_pca, y[split:])
        result["dim_control"] = control
    else:
        result["a_lin_null"] = 0.5
        result["a_lin_null_se"] = 0.0
        result["sample_size_curve"] = {}
        result["dim_control"] = {}

    return result


def train_probes_all_layers(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    concept: str,
    family: str,
    run_controls: bool = True,
    mlp_device: str = "cuda",
    n_workers: int | None = None,
) -> dict[int, ProbeResult]:
    """Train probes across all layers for one concept.

    CPU probes (liblinear + PCA-128) are parallelized across layers.
    MLP probes run sequentially on GPU after.

    Args:
        activations: Dict mapping layer index to (n_samples, d_model) arrays.
        labels: Binary labels (n_samples,).
        concept: Concept name.
        family: Family name.
        run_controls: Whether to run expensive controls.
        n_workers: Number of parallel workers. Defaults to min(n_layers, cpu_count - 2).
    """
    layers = sorted(activations.keys())
    n_layers = len(layers)

    if n_workers is None:
        n_workers = min(n_layers, max(1, mp.cpu_count() - 2))

    # ---- Step 1: CPU probes in parallel ----
    print(f"    Linear probes: {n_layers} layers × {n_workers} workers...", flush=True)

    cpu_results = {}
    # Use 'spawn' context to avoid inheriting torch's BLAS state (causes deadlocks with fork)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {}
        for layer_idx in layers:
            f = executor.submit(
                _train_cpu_probes_for_layer,
                activations[layer_idx], labels,
                layer_idx, concept, family,
                run_controls=run_controls,
            )
            futures[f] = layer_idx

        for f in as_completed(futures):
            layer_idx = futures[f]
            cpu_results[layer_idx] = f.result()

    print(f"    Linear probes done.", flush=True)

    # ---- Step 2: MLP probes sequentially on GPU ----
    print(f"    MLP probes: {n_layers} layers on {mlp_device}...", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlp_scores_by_layer = {}

    for layer_idx in layers:
        X = activations[layer_idx]
        nonlin_scores = []

        for train_idx, test_idx in skf.split(X, labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            if not _has_both_classes(y_train):
                continue

            mp_probe = MLPProbe(device=mlp_device, random_state=42)
            mp_probe.fit(X_train, y_train)
            nonlin_scores.append(mp_probe.score(X_test, y_test))

        if not nonlin_scores:
            nonlin_scores = [0.5]

        mlp_scores_by_layer[layer_idx] = nonlin_scores

    print(f"    MLP probes done.", flush=True)

    # ---- Step 3: Assemble ProbeResults ----
    results = {}
    for layer_idx in layers:
        cr = cpu_results[layer_idx]
        mlp_scores = mlp_scores_by_layer[layer_idx]

        a_nonlin = float(np.mean(mlp_scores))
        a_nonlin_se = float(np.std(mlp_scores) / np.sqrt(len(mlp_scores)))

        results[layer_idx] = ProbeResult(
            layer=layer_idx,
            concept=concept,
            family=family,
            a_lin=cr["a_lin"],
            a_lin_se=cr["a_lin_se"],
            a_nonlin=a_nonlin,
            a_nonlin_se=a_nonlin_se,
            probe_gap=a_nonlin - cr["a_lin"],
            a_lin_pca=cr["a_lin_pca"],
            a_lin_pca_se=cr["a_lin_pca_se"],
            weight_vectors=cr["weight_vectors"],
            direction_stability=cr["direction_stability"],
            a_lin_null=cr.get("a_lin_null", 0.5),
            a_lin_null_se=cr.get("a_lin_null_se", 0.0),
            sample_size_curve=cr.get("sample_size_curve", {}),
            dim_control=cr.get("dim_control", {}),
            best_C=cr["best_C"],
        )

    return results
