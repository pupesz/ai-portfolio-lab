import pandas as pd
import numpy as np

def simple_backtest(df: pd.DataFrame, pred_col: str = "pred_return", cost: float = 0.001):
    """
    Simple long/short backtest using predicted returns' sign as signal.
    """
    df = df.copy()
    df["signal"] = np.sign(df[pred_col])
    df["strategy_ret"] = df["signal"].shift(1) * df["returns"] - cost * df["signal"].diff().abs().fillna(0)
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    ann_factor = 252
    sharpe = np.sqrt(ann_factor) * df["strategy_ret"].mean() / df["strategy_ret"].std()
    mdd = (df["equity"].cummax() - df["equity"]).max() / df["equity"].cummax().max()
    cagr = df["equity"].iloc[-1] ** (ann_factor / len(df)) - 1
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDrawdown": float(mdd)}
