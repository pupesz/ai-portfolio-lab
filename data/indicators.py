import pandas as pd
import numpy as np

try:
    import ta  # make sure: pip install ta
except ImportError:
    ta = None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA, RSI, MACD and returns columns to a DataFrame containing 'close'.
    Only drops rows where the *core* features are NaN so we don't wipe everything.
    """
    out = df.copy()
    # sort by date just to be safe
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    # --- core indicators ---
    out["SMA_10"] = out["close"].rolling(10, min_periods=10).mean()
    out["SMA_50"] = out["close"].rolling(50, min_periods=50).mean()
    out["returns"] = out["close"].pct_change()

    # --- optional ta indicators ---
    out["RSI_14"] = np.nan
    out["MACD"] = np.nan
    out["MACD_signal"] = np.nan

    if ta is not None:
        try:
            rsi = ta.momentum.RSIIndicator(out["close"], window=14)
            out["RSI_14"] = rsi.rsi()
            macd = ta.trend.MACD(out["close"])
            out["MACD"] = macd.macd()
            out["MACD_signal"] = macd.macd_signal()
        except Exception as e:
            print("[add_indicators] Warning: ta indicators failed:", e)

    # Only drop rows where we *cannot* compute core features
    core_cols = ["SMA_10", "SMA_50", "returns"]
    out = out.dropna(subset=core_cols).reset_index(drop=True)

    return out
