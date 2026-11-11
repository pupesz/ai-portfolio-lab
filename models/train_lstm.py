"""
models/train_lstm.py
====================

Train the comparative LSTM on a single stock through the shared leak-free
core, using the IDENTICAL training protocol as the TCN (same loss, optimizer,
learning rate, batch size, early stopping). Saves the best checkpoint.

Usage:
    python -m models.train_lstm            # default symbol
    python -m models.train_lstm --symbol MSFT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiment.core import DEFAULT_LOOKBACK, prepare_stock
from experiment.models_factory import make_lstm
from experiment.trainer import train_model

CKPT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = ap.parse_args()

    data = prepare_stock(args.symbol, lookback=args.lookback)
    print(f"Training LSTM on {args.symbol}: "
          f"train={data.X_train.shape} val={data.X_val.shape} test={data.X_test.shape}")

    model, res = train_model(make_lstm, data, verbose=True)
    print(f"Best val loss {res.best_val_loss:.6f} at epoch {res.epochs_run}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = CKPT_DIR / f"lstm_{args.symbol}.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
