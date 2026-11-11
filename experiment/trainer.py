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
EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
PATIENCE = 25  # early-stopping patience on validation loss
GRAD_CLIP = 1.0  # max grad norm; stabilises LSTM training


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_run: int
    state_dict: dict


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def default_device() -> str:
    """Use CUDA when available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    grad_clip: float | None = GRAD_CLIP,
    device: str = None,
    seed: int = SEED,
    verbose: bool = False,
) -> Tuple[nn.Module, TrainResult]:
    """Train one model on one stock's prepared data with early stopping."""
    _set_seed(seed)
    if device is None:
        device = default_device()

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
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def predict(model: nn.Module, X: np.ndarray, device: str = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device.type
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(xb).squeeze(-1).cpu().numpy()
    return out.astype(np.float64)


def train_predict_multiseed(
    model_fn: Callable[[int], nn.Module],
    data: StockData,
    seeds: Tuple[int, ...] = (42, 1, 7),
    **train_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train the model once per seed, return the mean test prediction and the
    per-sample std across seeds (an uncertainty estimate). Averaging the
    predictions is an ensemble-mean that cuts single-seed noise.
    """
    preds = []
    for s in seeds:
        model, _ = train_model(model_fn, data, seed=s, **train_kwargs)
        preds.append(predict(model, data.X_test))
    P = np.stack(preds, axis=0)
    return P.mean(axis=0), P.std(axis=0)
