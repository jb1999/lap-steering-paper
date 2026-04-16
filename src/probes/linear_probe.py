"""Linear probe using logistic regression with L2 regularization."""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class LinearProbe:
    """Linear classifier probe for concept detection.

    Supports two modes:
    - full-dim with liblinear solver (handles high-dim/low-sample well)
    - PCA-reduced with lbfgs solver (focuses on high-variance subspace)
    """

    def __init__(
        self,
        Cs: int = 10,
        cv: int = 5,
        max_iter: int = 2000,
        random_state: int = 42,
        solver: str = "lbfgs",
        pca_dims: int | None = None,
    ):
        steps = [("scaler", StandardScaler())]

        if pca_dims is not None:
            steps.append(("pca", PCA(n_components=pca_dims, random_state=random_state)))
            # lbfgs works well after PCA reduction
            solver = "lbfgs"

        steps.append(("clf", LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            penalty="l2",
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            scoring="accuracy",
        )))

        self.pipeline = Pipeline(steps)
        self.pca_dims = pca_dims
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the probe.

        Args:
            X: Activations of shape (n_samples, d_model).
            y: Binary labels of shape (n_samples,).
        """
        # Guard against PCA dims exceeding sample count
        if self.pca_dims is not None and self.pca_dims >= min(X.shape):
            pca_step = self.pipeline.named_steps.get("pca")
            if pca_step is not None:
                pca_step.n_components = min(self.pca_dims, min(X.shape) - 1)

        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.pipeline.score(X, y)

    @property
    def weight_vectors(self) -> np.ndarray:
        """Return probe weight vectors (in original feature space).

        Returns shape (n_classes, d_model) for multi-class,
        or (1, d_model) for binary.
        """
        clf = self.pipeline.named_steps["clf"]
        W = clf.coef_  # (n_classes, n_features) or (1, n_features)

        if "pca" in self.pipeline.named_steps:
            pca = self.pipeline.named_steps["pca"]
            W = W @ pca.components_  # project back to full space

        scaler = self.pipeline.named_steps["scaler"]
        W = W / scaler.scale_[None, :]
        return W

    @property
    def best_C(self) -> float:
        return float(self.pipeline.named_steps["clf"].C_[0])
