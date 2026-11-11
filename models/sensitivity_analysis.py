"""
models/sensitivity_analysis.py
=============================

Two sensitivity experiments across the stock universe, comparing TCN and LSTM.

1. Decision-threshold sensitivity
   Fixed thresholds tau in {0.0025, 0.005, 0.010} plus a volatility-adapted
   threshold tau_i = 0.25 * sigma_5d_train (per stock). For each threshold we
   recompute trading + classification performance from the SAME trained-model
   predictions (training once per stock, re-thresholding is free).
   -> results/sensitivity_thresholds.csv

2. Lookback-window sensitivity
   Lookback L in {30, 60, 90}. Each L needs a fresh prepared dataset and fresh
   training (the input window changes), so this is the expensive sweep. We
   report test RMSE, Directional Accuracy and Sharpe.
   -> results/sensitivity_lookback.csv

Run:
    python -m models.sensitivity_analysis
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from experiment.core import (
    ALL_STOCKS,
    DEFAULT_LOOKBACK,
    RESULTS_DIR,
    backtest_from_signal,
    classification_metrics,
    directional_accuracy,
    ensure_results_dirs,
    prepare_stock,
    regression_metrics,
)
from experiment.models_factory import MODEL_FACTORIES
from experiment.trainer import predict, train_model, train_predict_multiseed

FIXED_TAUS = [0.0025, 0.005, 0.010]
LOOKBACKS = [30, 60, 90]
SEEDS = (42, 1, 7)


def _signal(pred: np.ndarray, tau: float) -> np.ndarray:
    s = np.zeros(len(pred))
    s[pred > tau] = 1.0
    s[pred < -tau] = -1.0
    return s


def threshold_sensitivity(
    symbols: List[str] = ALL_STOCKS,
    epochs: int = None,
    cache: dict = None,
) -> pd.DataFrame:
    """
    Re-threshold decisions WITHOUT retraining.

    If a prediction `cache` from evaluate_universe is supplied, reuse those
    multi-seed predictions (zero extra training). Otherwise train once per
    stock/model here. Either way, each threshold is applied to the SAME
    predictions, so the sweep isolates the threshold effect.
    """
    rows = []
    tk = {"epochs": epochs} if epochs else {}
    for sym in symbols:
        data = prepare_stock(sym, lookback=DEFAULT_LOOKBACK)
        y = data.y_test.astype(np.float64)
        vol_tau = 0.25 * data.sigma_5d_train
        tau_set = [("0.25%", 0.0025), ("0.50%", 0.005), ("1.00%", 0.010),
                   (f"vol({vol_tau*100:.2f}%)", vol_tau)]
        for model_name, factory in MODEL_FACTORIES.items():
            if cache is not None and sym in cache and f"{model_name}_pred" in cache[sym]:
                pred = cache[sym][f"{model_name}_pred"]
            else:
                pred, _ = train_predict_multiseed(factory, data, seeds=SEEDS, **tk)
            for tau_label, tau in tau_set:
                cls = classification_metrics(y, pred, tau)
                sig = _signal(pred, tau)
                bt = backtest_from_signal(sig, data.test_next_return)
                rows.append({
                    "Symbol": sym, "Model": model_name,
                    "Threshold": tau_label, "tau": tau,
                    "Overall_Acc": cls["Overall_Acc"],
                    "Buy_F1": cls["Buy_F1"], "Hold_F1": cls["Hold_F1"], "Avoid_F1": cls["Avoid_F1"],
                    "CAGR": bt["CAGR"], "Sharpe": bt["Sharpe"], "MaxDrawdown": bt["MaxDrawdown"],
                })
        print(f"  threshold sweep done: {sym}")
    df = pd.DataFrame(rows)
    ensure_results_dirs()
    out = RESULTS_DIR / "sensitivity_thresholds.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def lookback_sensitivity(symbols: List[str] = ALL_STOCKS, epochs: int = None) -> pd.DataFrame:
    """Re-train per lookback window (expensive; multi-seed averaged)."""
    rows = []
    tk = {"epochs": epochs} if epochs else {}
    for sym in symbols:
        for L in LOOKBACKS:
            data = prepare_stock(sym, lookback=L)
            y = data.y_test.astype(np.float64)
            for model_name, factory in MODEL_FACTORIES.items():
                pred, _ = train_predict_multiseed(factory, data, seeds=SEEDS, **tk)
                reg = regression_metrics(y, pred)
                dacc = directional_accuracy(y, pred)
                sig = _signal(pred, 0.005)
                bt = backtest_from_signal(sig, data.test_next_return)
                rows.append({
                    "Symbol": sym, "Model": model_name, "Lookback": L,
                    "RMSE": reg["RMSE"], "Directional_Acc": dacc, "Sharpe": bt["Sharpe"],
                })
        print(f"  lookback sweep done: {sym}")
    df = pd.DataFrame(rows)
    ensure_results_dirs()
    out = RESULTS_DIR / "sensitivity_lookback.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def main(symbols: List[str] = ALL_STOCKS, epochs: int = None):
    print("=== Threshold sensitivity ===")
    threshold_sensitivity(symbols, epochs=epochs)
    print("\n=== Lookback sensitivity ===")
    lookback_sensitivity(symbols, epochs=epochs)


if __name__ == "__main__":
    main()
