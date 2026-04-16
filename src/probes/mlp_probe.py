"""MLP probe: 2-layer neural network for nonlinear probing.

Supports both binary and multi-class classification.
"""

import numpy as np
import torch
import torch.nn as nn


class _MLPNet(nn.Module):
    def __init__(self, d_input: int, n_classes: int, d_hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes if n_classes > 2 else 1),
        )

    def forward(self, x):
        out = self.net(x)
        if self.n_classes <= 2:
            out = out.squeeze(-1)
        return out


class MLPProbe:
    """2-layer MLP probe for concept detection.

    Supports binary and multi-class. All training on GPU.
    """

    def __init__(
        self,
        d_hidden: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        device: str = "cuda",
        random_state: int = 42,
        n_classes: int | None = None,
    ):
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self._n_classes_override = n_classes
        self.model: _MLPNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_classes: int = 2

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    ) -> "MLPProbe":
        """Fit the MLP probe with early stopping."""
        torch.manual_seed(self.random_state)

        self._n_classes = self._n_classes_override or len(np.unique(y))

        if X_val is None:
            split = int(0.8 * len(X))
            X_val, y_val = X[split:], y[split:]
            X, y = X[:split], y[:split]

        X = self._normalize(X, fit=True)
        X_val = self._normalize(X_val)

        # Move all data to GPU upfront
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)

        if self._n_classes > 2:
            y_t = torch.tensor(y, dtype=torch.long, device=self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long, device=self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss()

        n_train = len(X_t)
        self.model = _MLPNet(
            X.shape[1], self._n_classes, self.d_hidden, self.dropout
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            for i in range(0, n_train, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self._normalize(X)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(X_t)
        if self._n_classes > 2:
            return logits.argmax(dim=-1).cpu().numpy()
        else:
            return (logits > 0).cpu().numpy().astype(int)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self._normalize(X)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.model(X_t)
        if self._n_classes > 2:
            return torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
            return np.stack([1 - probs, probs], axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))
