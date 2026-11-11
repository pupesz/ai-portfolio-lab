"""
generate_thesis_plots.py
========================

Publication-quality figures (300 DPI, unified academic style) for the thesis.
Reads the CSVs produced by the pipeline and, for equity curves, re-derives
per-day strategy returns from the shared core.

Figures saved to results/figures/:
  fig1_daily_volatility.png
  fig2_directional_accuracy_comparison.png
  fig3_cagr_comparison.png
  fig4_equity_curves.png
  fig5_class_accuracy_breakdown.png
  fig6_sensitivity_analysis.png

Run after the metrics + sensitivity CSVs exist:
    python -m generate_thesis_plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.core import (
    ALL_STOCKS,
    FIGURES_DIR,
    RESULTS_DIR,
    backtest_from_signal,
    buy_and_hold,
    load_combined,
    prepare_stock,
)

# ---- unified academic style ----
PALETTE = {
    "TCN": "#1f77b4",
    "LSTM": "#d62728",
    "Benchmark": "#2ca02c",
    "BuyHold": "#7f7f7f",
}
DPI = 300


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def fig1_daily_volatility():
    df = load_combined()
    vols = []
    for sym in ALL_STOCKS:
        sub = df[df["symbol"] == sym]
        vols.append((sym, float(sub["returns"].std())))
    vols.sort(key=lambda x: x[1], reverse=True)
    syms = [v[0] for v in vols]
    vals = [v[1] for v in vols]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(syms, vals, color="#1f77b4")
    ax.set_ylabel("Daily return std dev")
    ax.set_title("Daily Return Volatility by Stock")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_daily_volatility.png", dpi=DPI)
    plt.close(fig)


def _grouped_bar(ax, metrics, value_col, models, title, ylabel, hline=None):
    syms = ALL_STOCKS
    x = np.arange(len(syms))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = []
        for s in syms:
            row = metrics[(metrics["Symbol"] == s) & (metrics["Model"] == m)]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
        ax.bar(x + i * w, vals, w, label=m, color=PALETTE.get(m, None))
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(syms, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, ls="--", color="#666666", lw=1)
    ax.legend()


def fig2_directional_accuracy(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, metrics, "Directional_Acc", ["TCN", "LSTM", "Benchmark"],
                 "Directional Accuracy by Stock", "Directional accuracy", hline=0.50)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_directional_accuracy_comparison.png", dpi=DPI)
    plt.close(fig)


def fig3_cagr(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar(ax, metrics, "CAGR", ["TCN", "LSTM", "Benchmark", "BuyHold"],
                 "Backtest CAGR by Stock", "CAGR")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_cagr_comparison.png", dpi=DPI)
    plt.close(fig)


def fig4_equity_curves(cache):
    reps = ["AAPL", "AMD", "CAT", "GE"]
    reps = [s for s in reps if s in cache]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, sym in zip(axes.ravel(), reps):
        c = cache[sym]
        nxt = c["test_next_return"]
        dates = pd.to_datetime(c["test_dates"])
        for m in ["TCN", "LSTM"]:
            sig = np.zeros(len(c[f"{m}_pred"]))
            sig[c[f"{m}_pred"] > 0.005] = 1
            sig[c[f"{m}_pred"] < -0.005] = -1
            eq = _equity(sig, nxt)
            ax.plot(dates, eq, label=m, color=PALETTE[m], lw=1.5)
        # benchmark
        bsig = c["Benchmark_signal"]
        ax.plot(dates, _equity(bsig, c["Benchmark_next_return"]), label="Benchmark",
                color=PALETTE["Benchmark"], lw=1.2)
        ax.plot(dates, _equity(np.ones(len(nxt)), nxt), label="BuyHold",
                color=PALETTE["BuyHold"], lw=1.2, ls="--")
        ax.set_title(f"{sym} — test-set equity")
        ax.set_ylabel("Cumulative return (x)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_equity_curves.png", dpi=DPI)
    plt.close(fig)


def _equity(signal, nxt):
    nxt = np.nan_to_num(np.asarray(nxt, dtype=float), nan=0.0)
    pos = np.concatenate([[0.0], np.asarray(signal, dtype=float)[:-1]])
    return np.cumprod(1 + pos * nxt)


def fig5_class_accuracy(metrics):
    classes = ["Buy_Acc", "Hold_Acc", "Avoid_Acc"]
    labels = ["Buy", "Hold", "Avoid"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.35
    for i, m in enumerate(["TCN", "LSTM"]):
        vals = [float(metrics[metrics["Model"] == m][c].mean()) for c in classes]
        ax.bar(x + i * w, vals, w, label=m, color=PALETTE[m])
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean per-class accuracy")
    ax.set_title("Class Accuracy Breakdown (universe mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_class_accuracy_breakdown.png", dpi=DPI)
    plt.close(fig)


def fig6_sensitivity():
    thr_path = RESULTS_DIR / "sensitivity_thresholds.csv"
    lb_path = RESULTS_DIR / "sensitivity_lookback.csv"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if thr_path.exists():
        thr = pd.read_csv(thr_path)
        order = ["0.25%", "0.50%", "1.00%"]
        thr_fixed = thr[thr["Threshold"].isin(order)]
        for m in ["TCN", "LSTM"]:
            g = thr_fixed[thr_fixed["Model"] == m].groupby("Threshold")["Sharpe"].mean()
            g = g.reindex(order)
            axes[0].plot(order, g.values, marker="o", label=m, color=PALETTE[m])
        axes[0].set_title("(a) Threshold sensitivity")
        axes[0].set_xlabel("Decision threshold tau")
        axes[0].set_ylabel("Mean Sharpe")
        axes[0].legend()

    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        for m in ["TCN", "LSTM"]:
            g = lb[lb["Model"] == m].groupby("Lookback")["Sharpe"].mean()
            axes[1].plot(g.index.astype(str), g.values, marker="s", label=m, color=PALETTE[m])
        axes[1].set_title("(b) Lookback-window sensitivity")
        axes[1].set_xlabel("Lookback (trading days)")
        axes[1].set_ylabel("Mean Sharpe")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_sensitivity_analysis.png", dpi=DPI)
    plt.close(fig)


def generate_all(cache=None):
    _style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS_DIR / "model_comparison_metrics.csv")

    fig1_daily_volatility()
    fig2_directional_accuracy(metrics)
    fig3_cagr(metrics)
    fig5_class_accuracy(metrics)
    fig6_sensitivity()
    if cache is not None:
        fig4_equity_curves(cache)
    else:
        print("  (fig4 equity curves skipped: no prediction cache; run via run_pipeline)")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
