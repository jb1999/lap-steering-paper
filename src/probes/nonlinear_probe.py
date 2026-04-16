"""Nonlinear probe: MLP that transforms hidden states before unembedding.

Instead of classifying into N output classes (which fails when N is large),
this probe learns a nonlinear transformation in the hidden state space,
then applies the model's own unembedding matrix to project to token logits.

Linear probe:  h → W_unembed → logits
MLP probe:     h → MLP(h) → W_unembed → logits

The MLP learns what additional nonlinear processing the remaining layers
would have done. The gap between MLP and linear accuracy measures how
much nonlinear processing the concept requires.
"""

import numpy as np
import torch
import torch.nn as nn


class _ResidualMLP(nn.Module):
    """MLP that produces a corrected hidden state.

    Output is same dimension as input (d_model).
    Uses residual connection: output = input + MLP(input).
    """
    def __init__(self, d_model: int, d_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class NonlinearProbe:
    """MLP probe that operates in hidden state space, then unembeds.

    Args:
        unembed_weight: The model's unembedding matrix (vocab_size, d_model).
        layer_norm_weight: The model's final layer norm weight (d_model,).
        layer_norm_bias: The model's final layer norm bias (d_model,) or None.
    """

    def __init__(
        self,
        unembed_weight: torch.Tensor,
        layer_norm_weight: torch.Tensor,
        layer_norm_bias: torch.Tensor | None = None,
        d_hidden: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 5,
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.device = device
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

        # Store unembedding and layer norm (frozen, not trained)
        self.unembed = unembed_weight.to(device).float()  # (vocab, d_model)
        self.ln_weight = layer_norm_weight.to(device).float()
        self.ln_bias = layer_norm_bias.to(device).float() if layer_norm_bias is not None else None

        self.mlp: _ResidualMLP | None = None

    def _apply_layer_norm(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the model's final layer norm."""
        return nn.functional.layer_norm(
            h, (h.shape[-1],), weight=self.ln_weight, bias=self.ln_bias
        )

    def _to_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Apply layer norm + unembedding to get token logits."""
        h_normed = self._apply_layer_norm(h)
        return h_normed @ self.unembed.T  # (batch, vocab)

    def fit(
        self,
        H: np.ndarray,
        correct_token_ids: np.ndarray,
        H_val: np.ndarray | None = None,
        val_token_ids: np.ndarray | None = None,
    ) -> "NonlinearProbe":
        """Train the MLP probe.

        Args:
            H: Hidden states (n_samples, d_model).
            correct_token_ids: Correct token IDs (n_samples,).
        """
        if H_val is None:
            split = int(0.8 * len(H))
            H_val = H[split:]
            val_token_ids = correct_token_ids[split:]
            H = H[:split]
            correct_token_ids = correct_token_ids[:split]

        d_model = H.shape[1]
        self.mlp = _ResidualMLP(d_model, self.d_hidden, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            self.mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Move data to GPU
        H_t = torch.tensor(H, dtype=torch.float32, device=self.device)
        tids_t = torch.tensor(correct_token_ids, dtype=torch.long, device=self.device)
        H_val_t = torch.tensor(H_val, dtype=torch.float32, device=self.device)
        val_tids_t = torch.tensor(val_token_ids, dtype=torch.long, device=self.device)

        criterion = nn.CrossEntropyLoss()
        n_train = len(H_t)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.mlp.train()
            perm = torch.randperm(n_train, device=self.device)

            for i in range(0, n_train, self.batch_size):
                idx = perm[i:i + self.batch_size]
                h_batch = H_t[idx]
                tids_batch = tids_t[idx]

                # MLP transforms hidden state, then unembed
                h_transformed = self.mlp(h_batch)
                logits = self._to_logits(h_transformed)

                loss = criterion(logits, tids_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self.mlp.eval()
            with torch.no_grad():
                h_val_transformed = self.mlp(H_val_t)
                val_logits = self._to_logits(h_val_transformed)
                val_loss = criterion(val_logits, val_tids_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.mlp.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.mlp.load_state_dict(best_state)
            self.mlp.to(self.device)

        return self

    @torch.no_grad()
    def predict_top1(self, H: np.ndarray) -> np.ndarray:
        """Predict the top-1 token ID for each hidden state."""
        self.mlp.eval()
        H_t = torch.tensor(H, dtype=torch.float32, device=self.device)
        h_transformed = self.mlp(H_t)
        logits = self._to_logits(h_transformed)
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def score(self, H: np.ndarray, correct_token_ids: np.ndarray) -> float:
        """Top-1 accuracy: fraction where predicted token matches correct."""
        preds = self.predict_top1(H)
        return float(np.mean(preds == correct_token_ids))

    @torch.no_grad()
    def get_ranks(self, H: np.ndarray, correct_token_ids: np.ndarray) -> np.ndarray:
        """Rank of correct token in the MLP-transformed logits."""
        self.mlp.eval()
        H_t = torch.tensor(H, dtype=torch.float32, device=self.device)
        tids = torch.tensor(correct_token_ids, dtype=torch.long, device=self.device)

        h_transformed = self.mlp(H_t)
        logits = self._to_logits(h_transformed)

        ranks = []
        for i in range(len(H)):
            rank = int((logits[i] >= logits[i, tids[i]]).sum().item())
            ranks.append(rank)
        return np.array(ranks)
