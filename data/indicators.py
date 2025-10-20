import pandas as pd
import numpy as np

try:
    import ta  # pip install ta
except ImportError:
    ta = None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA, RSI, MACD and returns columns to a DataFrame containing 'close'."""
    out = df.copy()
    out["SMA_10"] = out["close"].rolling(10).mean()
    out["SMA_50"] = out["close"].rolling(50).mean()

    if ta is not None:
        out["RSI_14"] = ta.momentum.RSIIndicator(out["close"], 14).rsi()
        macd = ta.trend.MACD(out["close"])
        out["MACD"] = macd.macd()
        out["MACD_signal"] = macd.macd_signal()
    else:
        out["RSI_14"] = np.nan
        out["MACD"] = np.nan
        out["MACD_signal"] = np.nan

    out["returns"] = out["close"].pct_change()
    out = out.dropna().reset_index(drop=True)
    return out
