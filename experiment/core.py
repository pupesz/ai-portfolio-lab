"""
experiment/core.py
==================

Single, leak-free source of truth for the thesis experiments.

Everything downstream (TCN, LSTM, rule-based benchmark, buy-and-hold,
sensitivity analysis, figures, summary) reads its data through this module
so that every model and baseline is evaluated on the *exact* same:

  * target            : true 5-day forward return  Close(t+5)/Close(t) - 1
  * features          : the 6 indicator columns actually present in the data
  * split             : chronological 70% / 15% / 15% train/val/test
  * scaling           : StandardScaler fit STRICTLY on the 70% train split
                        (no look-ahead bias)

The old top-level scripts (train_tcn.py, evaluate_tcn.py, ...) are left
untouched so previous behaviour is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants shared across the whole experiment
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "combined_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# The 6 features that actually exist in combined_data.csv.
FEATURE_COLS: List[str] = ["SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_signal", "returns"]

# Stock universe (IT + Industrials), order fixed for reproducible reporting.
IT_STOCKS: List[str] = ["AAPL", "MSFT", "NVDA", "AMD", "ORCL"]
INDUSTRIAL_STOCKS: List[str] = ["CAT", "GE", "HON", "DE", "MMM"]
ALL_STOCKS: List[str] = IT_STOCKS + INDUSTRIAL_STOCKS

DEFAULT_LOOKBACK = 60
HORIZON = 5  # 5-day forward return target
SEED = 42

# Chronological split ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test ratio = 1 - TRAIN - VAL = 0.15

# 3-class decision band (used when no threshold override is given).
DEFAULT_TAU = 0.005  # +/- 0.5%

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Scaler (dependency-free StandardScaler so we don't require sklearn)
# ----------------------------------------------------------------------------
@dataclass
class StandardScaler:
    """Minimal StandardScaler: fit on train rows only, apply to all."""

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------------
# Per-stock prepared dataset
# ----------------------------------------------------------------------------
@dataclass
class StockData:
    """All arrays a model needs for one stock, with a leak-free split."""

    symbol: str
    lookback: int

    # Sequences (N, lookback, F) and targets (N,), already feature-scaled.
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Aligned test-set metadata for backtesting / labelling.
    test_dates: np.ndarray          # date at prediction time t
    test_close: np.ndarray          # Close(t)
    test_fwd_return: np.ndarray     # realized 5-day fwd return == y_test
    test_next_return: np.ndarray    # realized 1-day fwd return (for equity curves)

    # Training-set statistics (for volatility-adapted thresholds, etc.).
    sigma_5d_train: float           # std of 5-day fwd return on train split

    @property
    def n_features(self) -> int:
        return self.X_train.shape[2]


# ----------------------------------------------------------------------------
# Loading & sequence construction
# ----------------------------------------------------------------------------
_RAW_CACHE: Optional[pd.DataFrame] = None


def load_combined(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and lightly normalise the combined dataset (cached)."""
    global _RAW_CACHE
    if _RAW_CACHE is None:
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} not found. Run `python -m data.fetchdata` first."
            )
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        _RAW_CACHE = df
    return _RAW_CACHE


def _symbol_frame(symbol: str, data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = load_combined(data_path)
    sub = df[df["symbol"] == symbol].copy()
    if sub.empty:
        raise RuntimeError(f"No data for symbol {symbol}")
    sub = sub.sort_values("date").reset_index(drop=True)
    # Ensure the features and close are present and finite.
    needed = FEATURE_COLS + ["close"]
    sub = sub.dropna(subset=needed).reset_index(drop=True)
    return sub


def _true_5d_forward_return(close: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Close(t+h)/Close(t) - 1 ; NaN where t+h is out of range."""
    fwd = np.full(len(close), np.nan, dtype=np.float64)
    fwd[: len(close) - horizon] = close[horizon:] / close[: len(close) - horizon] - 1.0
    return fwd


def prepare_stock(
    symbol: str,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = HORIZON,
    data_path: Path = DATA_PATH,
) -> StockData:
    """
    Build leak-free, scaled sequences for one stock.

    Scaling rule: the StandardScaler is fit ONLY on the feature rows that
    fall in the 70% training split, then applied to val and test. The split
    is computed on the *sequence* index, and the training feature rows are
    exactly those used to build training sequences.
    """
    df = _symbol_frame(symbol, data_path)

    feats = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy()

    # True 5-day forward return target, aligned to time t.
    fwd_ret = _true_5d_forward_return(close, horizon)
    next_ret = _true_5d_forward_return(close, 1)  # 1-day fwd return for equity

    # Valid sequence start indices: need lookback history and a horizon future.
    # Sequence i uses rows [i, i+lookback) to predict return at t = i+lookback-1.
    n_rows = len(df)
    last_t = n_rows - horizon - 1            # last t with a realized fwd return
    starts = [i for i in range(0, last_t - lookback + 2) if (i + lookback - 1) <= last_t]
    starts = np.array(starts, dtype=int)
    if len(starts) < 30:
        raise RuntimeError(f"{symbol}: not enough rows for lookback={lookback}")

    # Chronological split on sequence order.
    n = len(starts)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Feature rows that belong to training sequences only -> fit scaler on these.
    train_starts = starts[:train_end]
    train_row_max = int(train_starts[-1] + lookback)  # exclusive
    scaler = StandardScaler().fit(feats[:train_row_max])
    feats_scaled = scaler.transform(feats)

    def build(idx_slice: np.ndarray):
        X, y = [], []
        for i in idx_slice:
            t = i + lookback - 1
            X.append(feats_scaled[i : i + lookback])
            y.append(fwd_ret[t])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    X_train, y_train = build(starts[:train_end])
    X_val, y_val = build(starts[train_end:val_end])
    X_test, y_test = build(starts[val_end:])

    # Test metadata aligned to prediction time t = i + lookback - 1.
    test_t = starts[val_end:] + lookback - 1
    test_dates = dates[test_t]
    test_close = close[test_t]
    test_fwd_return = fwd_ret[test_t]
    test_next_return = next_ret[test_t]

    # Training 5-day return volatility (for volatility-adapted thresholds).
    train_t = starts[:train_end] + lookback - 1
    sigma_5d_train = float(np.nanstd(fwd_ret[train_t]))

    return StockData(
        symbol=symbol,
        lookback=lookback,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_dates=test_dates,
        test_close=test_close,
        test_fwd_return=test_fwd_return,
        test_next_return=test_next_return,
        sigma_5d_train=sigma_5d_train,
    )


# ----------------------------------------------------------------------------
# Shared metric helpers
# ----------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def classify(returns: np.ndarray, tau: float) -> np.ndarray:
    """3-class labels: 1=Buy (>tau), 0=Hold ([-tau,tau]), -1=Avoid (<-tau)."""
    out = np.zeros(len(returns), dtype=int)
    out[returns > tau] = 1
    out[returns < -tau] = -1
    return out


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tau: float
) -> Dict[str, float]:
    """Overall + per-class accuracy and F1 for Buy / Hold / Avoid."""
    true_lbl = classify(y_true, tau)
    pred_lbl = classify(y_pred, tau)

    out: Dict[str, float] = {}
    out["Overall_Acc"] = float(np.mean(true_lbl == pred_lbl))

    names = {1: "Buy", 0: "Hold", -1: "Avoid"}
    for cls, name in names.items():
        tp = int(np.sum((pred_lbl == cls) & (true_lbl == cls)))
        fp = int(np.sum((pred_lbl == cls) & (true_lbl != cls)))
        fn = int(np.sum((pred_lbl != cls) & (true_lbl == cls)))
        support = int(np.sum(true_lbl == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # Per-class accuracy == recall on that class (fraction of true cls hit).
        out[f"{name}_Acc"] = recall
        out[f"{name}_F1"] = f1
        out[f"{name}_Support"] = float(support)
    return out


def backtest_from_signal(
    signal: np.ndarray,
    next_return: np.ndarray,
    cost: float = 0.0,
    trading_days: int = TRADING_DAYS,
    horizon: int = HORIZON,
) -> Dict[str, float]:
    """
    Daily backtest. `signal[t]` in {-1,0,1} is the position decided at t and
    held for the following step; `next_return[t]` is the realized 1-day return
    from t to t+1. CAGR is annualised from the number of test days.

    Returns CAGR, Sharpe, MaxDrawdown.
    """
    signal = np.asarray(signal, dtype=np.float64)
    r = np.asarray(next_return, dtype=np.float64)
    r = np.nan_to_num(r, nan=0.0)

    # Position taken at t affects return from t->t+1: shift signal by 1.
    pos = np.concatenate([[0.0], signal[:-1]])
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * r - cost * turnover

    equity = np.cumprod(1.0 + strat)
    if len(equity) == 0:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0}

    n_days = len(strat)
    total_growth = float(equity[-1])
    cagr = total_growth ** (trading_days / max(n_days, 1)) - 1.0 if total_growth > 0 else -1.0

    std = strat.std()
    sharpe = float(np.sqrt(trading_days) * strat.mean() / std) if std > 1e-12 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    mdd = float(np.max(drawdown)) if len(drawdown) else 0.0

    return {"CAGR": float(cagr), "Sharpe": sharpe, "MaxDrawdown": mdd}


def buy_and_hold(next_return: np.ndarray, trading_days: int = TRADING_DAYS) -> Dict[str, float]:
    """Passive long position over the whole test window."""
    signal = np.ones(len(next_return), dtype=np.float64)
    return backtest_from_signal(signal, next_return, cost=0.0, trading_days=trading_days)


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
