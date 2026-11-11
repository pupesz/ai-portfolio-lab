"""
models/evaluate_lstm.py
=======================

Evaluate the LSTM across the stock universe on the 15% test split and print a
per-stock metrics table. This is a thin wrapper that reuses the shared
evaluation driver so the LSTM is scored identically to the TCN.

Usage:
    python -m models.evaluate_lstm
"""
from __future__ import annotations

from experiment.evaluate_all import evaluate_universe


def main():
    res = evaluate_universe(verbose=True)
    df = res["metrics"]
    lstm = df[df["Model"] == "LSTM"][
        ["Symbol", "Sector", "MAE", "RMSE", "Directional_Acc", "CAGR", "Sharpe", "MaxDrawdown"]
    ]
    print("\n===== LSTM test-set metrics =====")
    print(lstm.to_string(index=False))


if __name__ == "__main__":
    main()
