"""Failure mode clustering (Experiment 3).

Clusters failed prompts by LAP signature and maps clusters
to the regression failure taxonomy.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Mapping from regression failure types to expected LAP signatures
FAILURE_TAXONOMY = {
    "superposition": {
        "description": "High condition number → multicollinearity / interference",
        "dominant_feature": "condition_number",
        "expected_high": ["condition_number"],
        "expected_low": [],
    },
    "nonlinear_residual": {
        "description": "Large probe gap → unresolved nonlinearity",
        "dominant_feature": "probe_gap",
        "expected_high": ["probe_gap"],
        "expected_low": [],
    },
    "omitted_features": {
        "description": "Low R²_sparse but high probe accuracy → missing SAE features",
        "dominant_feature": "r2_sparse",
        "expected_high": ["a_lin"],
        "expected_low": ["r2_sparse"],
    },
    "unstable_basis": {
        "description": "Low direction stability → basis instability",
        "dominant_feature": "direction_stability",
        "expected_high": [],
        "expected_low": ["direction_stability"],
    },
    "distribution_edge": {
        "description": "High Mahalanobis distance → adversarial / OOD",
        "dominant_feature": None,  # Computed separately
        "expected_high": [],
        "expected_low": [],
    },
}


@dataclass
class ClusterInfo:
    """Information about one failure cluster."""
    cluster_id: int
    n_prompts: int
    centroid: np.ndarray  # In LAP feature space
    dominant_features: list[str]  # Features with highest deviation from overall mean
    taxonomy_match: str  # Best matching failure type
    taxonomy_score: float  # How well it matches


@dataclass
class ClusteringResult:
    """Result of failure mode clustering."""
    n_clusters: int
    silhouette_score: float
    cluster_labels: np.ndarray  # (n_failed_prompts,)
    clusters: list[ClusterInfo]
    pca_coords: np.ndarray  # (n_failed_prompts, 2) for visualization
    pca_model: PCA
    feature_names: list[str]

    # Stability across k
    silhouette_by_k: dict[int, float] = field(default_factory=dict)


class FailureClustering:
    """Cluster interpretability failures by LAP signature."""

    def __init__(
        self,
        k_range: tuple[int, ...] = (2, 3, 4, 5),
        random_state: int = 42,
    ):
        self.k_range = k_range
        self.random_state = random_state

    def cluster(
        self,
        X_failed: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ClusteringResult:
        """Cluster failed prompts by LAP signature.

        Args:
            X_failed: LAP features for failed prompts (n_failed, 10).
            feature_names: Names of the 10 features.

        Returns:
            ClusteringResult with cluster assignments and analysis.
        """
        from src.metrics.lap import LAPVector
        if feature_names is None:
            feature_names = LAPVector.feature_names()

        # Remove NaN features
        valid_cols = ~np.any(np.isnan(X_failed), axis=0)
        X_clean = X_failed[:, valid_cols]
        clean_names = [n for n, v in zip(feature_names, valid_cols) if v]

        if len(X_clean) < max(self.k_range) + 1:
            # Too few samples
            return ClusteringResult(
                n_clusters=1,
                silhouette_score=0.0,
                cluster_labels=np.zeros(len(X_failed), dtype=int),
                clusters=[],
                pca_coords=np.zeros((len(X_failed), 2)),
                pca_model=PCA(n_components=2),
                feature_names=feature_names,
            )

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Find best k by silhouette score
        silhouette_by_k = {}
        best_k = self.k_range[0]
        best_score = -1

        for k in self.k_range:
            if k >= len(X_scaled):
                continue
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                silhouette_by_k[k] = score
                if score > best_score:
                    best_score = score
                    best_k = k

        # Final clustering with best k
        km = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        pca_coords = pca.fit_transform(X_scaled)

        # Analyze clusters
        overall_mean = X_scaled.mean(axis=0)
        clusters = []
        for c in range(best_k):
            mask = labels == c
            centroid = X_scaled[mask].mean(axis=0)

            # Dominant features: highest absolute deviation from overall mean
            deviations = np.abs(centroid - overall_mean)
            top_feat_idx = np.argsort(deviations)[::-1][:3]
            dominant_features = [clean_names[i] for i in top_feat_idx]

            # Match to taxonomy
            taxonomy_match, taxonomy_score = self._match_taxonomy(
                centroid, overall_mean, clean_names
            )

            clusters.append(ClusterInfo(
                cluster_id=c,
                n_prompts=int(mask.sum()),
                centroid=scaler.inverse_transform(centroid.reshape(1, -1))[0],
                dominant_features=dominant_features,
                taxonomy_match=taxonomy_match,
                taxonomy_score=taxonomy_score,
            ))

        return ClusteringResult(
            n_clusters=best_k,
            silhouette_score=best_score,
            cluster_labels=labels,
            clusters=clusters,
            pca_coords=pca_coords,
            pca_model=pca,
            feature_names=feature_names,
            silhouette_by_k=silhouette_by_k,
        )

    def _match_taxonomy(
        self,
        centroid: np.ndarray,
        overall_mean: np.ndarray,
        feature_names: list[str],
    ) -> tuple[str, float]:
        """Match a cluster centroid to the failure taxonomy.

        Returns (best_match_name, score).
        """
        best_match = "unclassified"
        best_score = 0.0

        deviations = centroid - overall_mean  # Signed deviations

        for fail_type, spec in FAILURE_TAXONOMY.items():
            if spec["dominant_feature"] is None:
                continue

            score = 0.0
            n_checks = 0

            for feat in spec["expected_high"]:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    score += max(0, deviations[idx])  # Positive deviation = high
                    n_checks += 1

            for feat in spec["expected_low"]:
                if feat in feature_names:
                    idx = feature_names.index(feat)
                    score += max(0, -deviations[idx])  # Negative deviation = low
                    n_checks += 1

            if n_checks > 0:
                score /= n_checks
            if score > best_score:
                best_score = score
                best_match = fail_type

        return best_match, best_score
