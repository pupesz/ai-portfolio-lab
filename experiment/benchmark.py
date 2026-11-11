"""
experiment/benchmark.py
=======================

Rule-based technical-indicator benchmark (SMA / MACD / RSI) evaluated on the
*same* 15% chronological test slice as the deep models, plus the passive
Buy-and-Hold baseline.

The benchmark produces, for each test day t, a discrete position in {-1,0,1}
(Avoid / Hold / Buy) from classic indicator rules. It is a trading strategy,
not a return regressor, so it has no MAE/RMSE -- only directional,
classification and backtest metrics are meaningful for it.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from experiment.core import (
    FEATURE_COLS,
    HORIZON,
    TRAIN_RATIO,
    VAL_RATIO,
    _symbol_frame,
    DEFAULT_LOOKBACK,
)


def _rule_signal(row) -> int:
    """
    Classic long/flat/avoid rule from SMA, MACD and RSI.

    Buy  (+1): short SMA above long SMA (uptrend) AND MACD above its signal
               AND RSI not overbought (< 70).
    Avoid(-1): short SMA below long SMA (downtrend) AND MACD below its signal
               AND RSI not oversold (> 30).
    Hold ( 0): everything else (mixed signals).
    """
    sma_up = row["SMA_10"] > row["SMA_50"]
    macd_up = row["MACD"] > row["MACD_signal"]
    rsi = row["RSI_14"]

    if sma_up and macd_up and rsi < 70:
        return 1
    if (not sma_up) and (not macd_up) and rsi > 30:
        return -1
    return 0


def benchmark_test_frame(symbol: str, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """
    Return a dataframe over the SAME test rows the deep models use, with:
      signal, fwd_return (5-day), next_return (1-day), date, close.
    """
    df = _symbol_frame(symbol)
    close = df["close"].to_numpy(dtype=np.float64)
    n_rows = len(df)

    # Reconstruct the identical sequence index + split used in core.prepare_stock.
    horizon = HORIZON
    last_t = n_rows - horizon - 1
    starts = np.array(
        [i for i in range(0, last_t - lookback + 2) if (i + lookback - 1) <= last_t],
        dtype=int,
    )
    n = len(starts)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    test_t = starts[val_end:] + lookback - 1  # prediction-time rows

    fwd = np.full(n_rows, np.nan)
    fwd[: n_rows - horizon] = close[horizon:] / close[: n_rows - horizon] - 1.0
    nxt = np.full(n_rows, np.nan)
    nxt[: n_rows - 1] = close[1:] / close[: n_rows - 1] - 1.0

    sub = df.iloc[test_t].copy().reset_index(drop=True)
    sub["signal"] = sub.apply(_rule_signal, axis=1).astype(int)
    sub["fwd_return"] = fwd[test_t]
    sub["next_return"] = nxt[test_t]
    return sub[["date", "close", "signal", "fwd_return", "next_return"] + FEATURE_COLS]
