# models/recommend.py
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from data.fetchdata import get_stock_data
from data.indicators import add_indicators
from models.tcn_model import TCN

# If you defined FEATURE_COLS in data/dataset.py, you can import it instead of redefining:
try:
    from data.dataset import FEATURE_COLS
except ImportError:
    FEATURE_COLS = ["SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_signal", "returns"]

# --- config (use same as in training) ---
LOOKBACK = 60
HORIZON = 5

# decision thresholds (tune based on backtests if you want)
BUY_THRESHOLD = 0.005   # +0.5% expected over next 5 days
HOLD_THRESHOLD = -0.005 # below this, it's clearly bad

SYMBOLS: List[str] = ["NVDA", "AAPL", "MSFT", "CAT", "HON", "DE"]

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pt"


def load_model(n_features: int) -> TCN:
    model = TCN(n_features=n_features)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_latest_sequence(symbol: str) -> np.ndarray:
    """Fetch latest data for symbol, add indicators, and return the last LOOKBACK window."""
    df = get_stock_data(symbol)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)

    if len(df) < LOOKBACK:
        raise RuntimeError(f"Not enough data for {symbol} after indicators (need {LOOKBACK}, have {len(df)})")

    window = df.tail(LOOKBACK)[FEATURE_COLS].values  # shape: (LOOKBACK, F)
    return window.astype("float32")


def classify_decision(pred_return: float) -> str:
    """Convert predicted next-5d return into a human decision."""
    if pred_return > BUY_THRESHOLD:
        return "BUY"
    elif pred_return < HOLD_THRESHOLD:
        return "AVOID / POSSIBLE SELL"
    else:
        return "HOLD / NO ACTION"


def main():
    # Prepare first symbol to infer number of features
    first_symbol = SYMBOLS[0]
    first_window = get_latest_sequence(first_symbol)
    n_features = first_window.shape[1]

    model = load_model(n_features=n_features)

    rows = []
    for sym in SYMBOLS:
        try:
            window = get_latest_sequence(sym)
        except RuntimeError as e:
            print(f"[{sym}] Skipping: {e}")
            continue

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, LOOKBACK, F)
        with torch.no_grad():
            pred = model(x).item()  # predicted next-5d return

        decision = classify_decision(pred)
        rows.append((sym, pred, decision))
        print(f"{sym}: predicted 5d return = {pred:+.4%} → {decision}")

    if rows:
        out_df = pd.DataFrame(rows, columns=["Symbol", "Pred_5d_Return", "Decision"])
        out_path = Path(__file__).resolve().parent.parent / "recommendations.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\n📁 Detailed recommendations saved to: {out_path}")


if __name__ == "__main__":
    main()
