"""
experiment/evaluate_all.py
==========================

Train per-stock TCN and LSTM through the shared leak-free core, evaluate them
alongside the rule-based benchmark and Buy-and-Hold on the identical 15% test
split, and export the full comparison table.

Outputs:
  results/model_comparison_metrics.csv   (one row per stock x model)
  returns an in-memory dict of predictions for reuse by the figure script.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from experiment.core import (
    ALL_STOCKS,
    DEFAULT_LOOKBACK,
    DEFAULT_TAU,
    IT_STOCKS,
    RESULTS_DIR,
    backtest_from_signal,
    buy_and_hold,
    classification_metrics,
    directional_accuracy,
    ensure_results_dirs,
    prepare_stock,
    regression_metrics,
)
from experiment.benchmark import benchmark_test_frame
from experiment.models_factory import MODEL_FACTORIES
from experiment.trainer import predict, train_model

COST = 0.0  # frictionless by default; set >0 to add transaction cost


def _signal_from_pred(pred: np.ndarray, tau: float) -> np.ndarray:
    sig = np.zeros(len(pred), dtype=np.float64)
    sig[pred > tau] = 1.0
    sig[pred < -tau] = -1.0
    return sig


def evaluate_universe(
    symbols: List[str] = ALL_STOCKS,
    lookback: int = DEFAULT_LOOKBACK,
    tau: float = DEFAULT_TAU,
    epochs: int = None,
    verbose: bool = True,
) -> Dict:
    """Run the full comparison; return rows + per-stock prediction cache."""
    ensure_results_dirs()
    rows: List[dict] = []
    cache: Dict[str, dict] = {}  # symbol -> model/benchmark predictions & series

    train_kwargs = {}
    if epochs is not None:
        train_kwargs["epochs"] = epochs

    for sym in symbols:
        if verbose:
            print(f"\n=== {sym} ===")
        data = prepare_stock(sym, lookback=lookback)
        cache[sym] = {
            "test_dates": data.test_dates,
            "test_next_return": data.test_next_return,
            "y_test": data.y_test.astype(np.float64),
            "sigma_5d_train": data.sigma_5d_train,
            "sector": "IT" if sym in IT_STOCKS else "Industrial",
        }

        # ---- deep models (TCN, LSTM) ----
        for model_name, factory in MODEL_FACTORIES.items():
            model, tr = train_model(factory, data, verbose=False, **train_kwargs)
            pred = predict(model, data.X_test)
            cache[sym][f"{model_name}_pred"] = pred

            reg = regression_metrics(data.y_test.astype(np.float64), pred)
            dacc = directional_accuracy(data.y_test.astype(np.float64), pred)
            cls = classification_metrics(data.y_test.astype(np.float64), pred, tau)
            sig = _signal_from_pred(pred, tau)
            bt = backtest_from_signal(sig, data.test_next_return, cost=COST)

            rows.append(
                {
                    "Symbol": sym,
                    "Sector": cache[sym]["sector"],
                    "Model": model_name,
                    **reg,
                    "Directional_Acc": dacc,
                    **cls,
                    "CAGR": bt["CAGR"],
                    "Sharpe": bt["Sharpe"],
                    "MaxDrawdown": bt["MaxDrawdown"],
                }
            )
            if verbose:
                print(
                    f"  {model_name:5s} RMSE={reg['RMSE']:.4f} "
                    f"DirAcc={dacc*100:5.1f}% CAGR={bt['CAGR']*100:6.1f}% "
                    f"Sharpe={bt['Sharpe']:.2f}"
                )

        # ---- rule-based benchmark ----
        bench = benchmark_test_frame(sym, lookback=lookback)
        bench_sig = bench["signal"].to_numpy(dtype=np.float64)
        bench_next = bench["next_return"].to_numpy(dtype=np.float64)
        bench_fwd = bench["fwd_return"].to_numpy(dtype=np.float64)
        cache[sym]["Benchmark_signal"] = bench_sig
        cache[sym]["Benchmark_next_return"] = bench_next

        # Benchmark directional acc: sign of position vs realized fwd return.
        valid = ~np.isnan(bench_fwd)
        b_dacc = float(
            np.mean(np.sign(bench_sig[valid]) == np.sign(bench_fwd[valid]))
        ) if valid.any() else 0.0
        # Classification: treat benchmark signal as its class prediction.
        b_cls = _benchmark_classification(bench_sig[valid], bench_fwd[valid], tau)
        b_bt = backtest_from_signal(bench_sig, bench_next, cost=COST)
        rows.append(
            {
                "Symbol": sym,
                "Sector": cache[sym]["sector"],
                "Model": "Benchmark",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "Directional_Acc": b_dacc,
                **b_cls,
                "CAGR": b_bt["CAGR"],
                "Sharpe": b_bt["Sharpe"],
                "MaxDrawdown": b_bt["MaxDrawdown"],
            }
        )

        # ---- buy & hold ----
        bh = buy_and_hold(data.test_next_return)
        rows.append(
            {
                "Symbol": sym,
                "Sector": cache[sym]["sector"],
                "Model": "BuyHold",
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "Directional_Acc": np.nan,
                "CAGR": bh["CAGR"],
                "Sharpe": bh["Sharpe"],
                "MaxDrawdown": bh["MaxDrawdown"],
            }
        )
        if verbose:
            print(
                f"  Bench DirAcc={b_dacc*100:5.1f}% CAGR={b_bt['CAGR']*100:6.1f}% | "
                f"BuyHold CAGR={bh['CAGR']*100:6.1f}%"
            )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "model_comparison_metrics.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"\nSaved {out}")
    return {"metrics": df, "cache": cache}


def _benchmark_classification(sig, fwd, tau):
    """Per-class accuracy/F1 for the benchmark, labels from the signal itself."""
    true = np.zeros(len(fwd), dtype=int)
    true[fwd > tau] = 1
    true[fwd < -tau] = -1
    pred = sig.astype(int)
    out = {"Overall_Acc": float(np.mean(true == pred)) if len(true) else 0.0}
    for cls, name in {1: "Buy", 0: "Hold", -1: "Avoid"}.items():
        tp = int(np.sum((pred == cls) & (true == cls)))
        fp = int(np.sum((pred == cls) & (true != cls)))
        fn = int(np.sum((pred != cls) & (true == cls)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[f"{name}_Acc"] = rec
        out[f"{name}_F1"] = f1
        out[f"{name}_Support"] = float(int(np.sum(true == cls)))
    return out


if __name__ == "__main__":
    evaluate_universe()
