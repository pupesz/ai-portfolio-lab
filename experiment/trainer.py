"""
experiment/trainer.py
=====================

Shared training + prediction routine used by BOTH the TCN and the LSTM, so
the two models differ only in architecture, never in training protocol.

Protocol (identical for both models):
  * loss            : MSELoss
  * optimizer       : AdamW (lr, weight_decay configurable)
  * batching        : mini-batches of BATCH_SIZE
  * early stopping  : monitor validation loss, patience epochs, restore best
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiment.core import SEED, StockData

# ----------------------------------------------------------------------------
# Default training hyperparameters (shared by TCN and LSTM)
# ----------------------------------------------------------------------------
EPOCHS = 60
LR = 5e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early-stopping patience on validation loss


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_run: int
    state_dict: dict


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _batch_iter(X: torch.Tensor, y: torch.Tensor, bs: int):
    for i in range(0, len(X), bs):
        yield X[i : i + bs], y[i : i + bs]


def train_model(
    model_fn: Callable[[int], nn.Module],
    data: StockData,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: str = "cpu",
    seed: int = SEED,
    verbose: bool = False,
) -> Tuple[nn.Module, TrainResult]:
    """Train one model on one stock's prepared data with early stopping."""
    _set_seed(seed)

    model = model_fn(data.n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(data.X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(data.y_train, dtype=torch.float32, device=device)
    X_va = torch.tensor(data.X_val, dtype=torch.float32, device=device)
    y_va = torch.tensor(data.y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in _batch_iter(X_tr, y_tr, batch_size):
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va).squeeze(-1)
            val_loss = float(criterion(val_pred, y_va).item())

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:03d} | val {val_loss:.6f} | best {best_val:.6f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, TrainResult(best_val_loss=best_val, epochs_run=best_epoch, state_dict=best_state)


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(xb).squeeze(-1).cpu().numpy()
    return out.astype(np.float64)
