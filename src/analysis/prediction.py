"""LAP → tool-success prediction pipeline.

Implements ridge/logistic regression with 5-fold CV,
baseline comparisons (A-E), and leave-one-concept-out evaluation.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr, permutation_test


@dataclass
class PredictionResult:
    """Result of LAP → tool-success prediction."""
    predictor_name: str
    tool_name: str

    # Regression metrics
    r2_cv: float = 0.0  # Cross-validated R²
    r2_cv_se: float = 0.0

    # Classification metrics
    auc_cv: float = 0.0  # Cross-validated AUC
    auc_cv_se: float = 0.0

    # Correlation
    spearman_rho: float = 0.0
    spearman_p: float = 1.0

    # Per-fold details
    r2_folds: list[float] = field(default_factory=list)
    auc_folds: list[float] = field(default_factory=list)


@dataclass
class BaselineComparison:
    """Comparison of full LAP vs. baselines."""
    tool_name: str
    full_lap: PredictionResult | None = None
    probe_only: PredictionResult | None = None  # Baseline A
    concept_depth_only: PredictionResult | None = None  # Baseline B
    geometry_only: PredictionResult | None = None  # Baseline C
    decomposition_only: PredictionResult | None = None  # Baseline D
    random: PredictionResult | None = None  # Baseline E

    # Statistical test: full LAP vs. probe-only
    permutation_p_value: float = 1.0
    lap_beats_probe: bool = False


class LAPPredictor:
    """Predict tool success from LAP features."""

    # Feature indices for each subset
    PROBE_FEATURES = [0, 1, 2]  # a_lin, a_nonlin, probe_gap
    GEOMETRY_FEATURES = [3, 4, 5, 6]  # ρ, κ_32, κ_128, cond
    DECOMPOSITION_FEATURES = [7, 8, 9]  # R²_sparse, δ, λ
    ALL_FEATURES = list(range(10))

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state

    def predict_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_mask: list[int] | None = None,
        predictor_name: str = "full_lap",
        tool_name: str = "unknown",
    ) -> PredictionResult:
        """Cross-validated prediction of tool success.

        Args:
            X: LAP feature matrix (n_samples, 10).
            y: Tool-success metric (n_samples,). Continuous for regression.
            feature_mask: Which features to use (indices). None = all.
            predictor_name: Name for logging.
            tool_name: Tool name for logging.
        """
        if feature_mask is not None:
            X = X[:, feature_mask]

        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < self.n_folds * 2:
            return PredictionResult(predictor_name=predictor_name, tool_name=tool_name)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        r2_folds = []
        auc_folds = []
        all_preds = np.zeros(len(y))
        all_true = np.zeros(len(y))

        # Binarize for AUC (median split)
        y_binary = (y > np.median(y)).astype(int)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_bin_train, y_bin_test = y_binary[train_idx], y_binary[test_idx]

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Ridge regression
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X_train_s, y_train)
            y_pred = ridge.predict(X_test_s)
            r2 = r2_score(y_test, y_pred)
            r2_folds.append(r2)

            all_preds[test_idx] = y_pred
            all_true[test_idx] = y_test

            # Logistic regression for AUC
            if len(np.unique(y_bin_train)) > 1 and len(np.unique(y_bin_test)) > 1:
                try:
                    lr = LogisticRegressionCV(
                        Cs=10, cv=3, max_iter=1000,
                        random_state=self.random_state,
                    )
                    lr.fit(X_train_s, y_bin_train)
                    y_prob = lr.predict_proba(X_test_s)[:, 1]
                    auc = roc_auc_score(y_bin_test, y_prob)
                    auc_folds.append(auc)
                except Exception:
                    pass

        # Spearman correlation on held-out predictions
        rho, p = spearmanr(all_true, all_preds)

        return PredictionResult(
            predictor_name=predictor_name,
            tool_name=tool_name,
            r2_cv=float(np.mean(r2_folds)),
            r2_cv_se=float(np.std(r2_folds) / np.sqrt(len(r2_folds))),
            auc_cv=float(np.mean(auc_folds)) if auc_folds else 0.0,
            auc_cv_se=float(np.std(auc_folds) / np.sqrt(len(auc_folds))) if auc_folds else 0.0,
            spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
            spearman_p=float(p) if not np.isnan(p) else 1.0,
            r2_folds=r2_folds,
            auc_folds=auc_folds,
        )

    def compare_baselines(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tool_name: str,
    ) -> BaselineComparison:
        """Compare full LAP vs. all baselines (A-E).

        Args:
            X: Full LAP feature matrix (n_samples, 10).
            y: Tool-success metric (n_samples,).
            tool_name: Tool name.
        """
        full = self.predict_cv(X, y, None, "full_lap", tool_name)
        probe = self.predict_cv(X, y, self.PROBE_FEATURES, "probe_only", tool_name)
        geometry = self.predict_cv(X, y, self.GEOMETRY_FEATURES, "geometry_only", tool_name)
        decomp = self.predict_cv(X, y, self.DECOMPOSITION_FEATURES, "decomposition_only", tool_name)

        # Random baseline: shuffled LAP
        rng = np.random.RandomState(self.random_state)
        X_shuffled = X.copy()
        for col in range(X_shuffled.shape[1]):
            rng.shuffle(X_shuffled[:, col])
        random = self.predict_cv(X_shuffled, y, None, "random", tool_name)

        # Concept depth baseline: just the layer index as feature
        # (This is computed at the experiment level, not here)

        # Permutation test: full LAP vs probe-only
        p_val = self._permutation_test_r2(X, y, self.PROBE_FEATURES)

        return BaselineComparison(
            tool_name=tool_name,
            full_lap=full,
            probe_only=probe,
            geometry_only=geometry,
            decomposition_only=decomp,
            random=random,
            permutation_p_value=p_val,
            lap_beats_probe=(full.r2_cv > probe.r2_cv and p_val < 0.01),
        )

    def _permutation_test_r2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        baseline_features: list[int],
        n_permutations: int = 1000,
    ) -> float:
        """Paired permutation test: full LAP R² vs. baseline R².

        Tests whether the difference in R² is significant.
        """
        # Remove NaN
        valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        X = X[valid]
        y = y[valid]

        if len(X) < 20:
            return 1.0

        # Get per-fold R² differences
        full_result = self.predict_cv(X, y, None, "full", "test")
        base_result = self.predict_cv(X, y, baseline_features, "base", "test")

        observed_diff = full_result.r2_cv - base_result.r2_cv

        # Permutation: shuffle assignment of features to "full" vs "baseline"
        rng = np.random.RandomState(self.random_state)
        count_greater = 0

        for _ in range(n_permutations):
            # Randomly swap feature subsets
            perm_features = rng.permutation(X.shape[1])
            perm_X = X[:, perm_features]

            perm_full = self.predict_cv(perm_X, y, None, "perm_full", "test")
            perm_base = self.predict_cv(perm_X, y, baseline_features, "perm_base", "test")

            perm_diff = perm_full.r2_cv - perm_base.r2_cv
            if perm_diff >= observed_diff:
                count_greater += 1

        return (count_greater + 1) / (n_permutations + 1)

    def leave_one_concept_out(
        self,
        X_by_family: dict[str, np.ndarray],
        y_by_family: dict[str, np.ndarray],
        tool_name: str,
    ) -> dict[str, PredictionResult]:
        """Leave-one-concept-family-out evaluation.

        Train on 4 families, test on the 5th.

        Args:
            X_by_family: Dict mapping family name to LAP features.
            y_by_family: Dict mapping family name to tool success metrics.
            tool_name: Tool name.

        Returns:
            Dict mapping held-out family to PredictionResult.
        """
        families = sorted(X_by_family.keys())
        results = {}

        for held_out in families:
            # Train set: all other families
            X_train = np.vstack([X_by_family[f] for f in families if f != held_out])
            y_train = np.concatenate([y_by_family[f] for f in families if f != held_out])

            X_test = X_by_family[held_out]
            y_test = y_by_family[held_out]

            # Remove NaN
            valid_train = ~np.any(np.isnan(X_train), axis=1) & ~np.isnan(y_train)
            valid_test = ~np.any(np.isnan(X_test), axis=1) & ~np.isnan(y_test)
            X_train, y_train = X_train[valid_train], y_train[valid_train]
            X_test, y_test = X_test[valid_test], y_test[valid_test]

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
            ridge.fit(X_train_s, y_train)
            y_pred = ridge.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            rho, p = spearmanr(y_test, y_pred)

            results[held_out] = PredictionResult(
                predictor_name=f"loco_{held_out}",
                tool_name=tool_name,
                r2_cv=r2,
                spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
                spearman_p=float(p) if not np.isnan(p) else 1.0,
            )

        return results

    def compute_lap_index(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, PCA]:
        """Compute LAP index as PC1 of standardized metrics.

        For visualization only, not the primary predictive object.

        Args:
            X_train: Training LAP features (n_train, 10).
            X_test: Optional test LAP features.

        Returns:
            (train_index, test_index, pca_model)
        """
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        pca = PCA(n_components=1, random_state=self.random_state)
        train_index = pca.fit_transform(X_train_s).ravel()

        test_index = None
        if X_test is not None:
            X_test_s = scaler.transform(X_test)
            test_index = pca.transform(X_test_s).ravel()

        return train_index, test_index, pca
