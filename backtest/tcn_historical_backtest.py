# backtest/tcn_historical_backtest.py
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from data.indicators import add_indicators
from models.tcn_model import TCN
from backtest.backtest import simple_backtest

# If you already have FEATURE_COLS in data/dataset.py, import it:
try:
    from data.dataset import FEATURE_COLS
except ImportError:
    FEATURE_COLS = ["SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_signal", "returns"]

# ---------------- CONFIG ----------------
LOOKBACK = 60          # same as training
HORIZON = 5            # same as training target

# thresholds for classification (tune as you like)
BUY_THRESHOLD = 0.005    # > +0.5% over HORIZON days → BUY
HOLD_THRESHOLD = -0.005  # < -0.5% over HORIZON days → AVOID, else HOLD

SYMBOLS: List[str] = ["NVDA", "AAPL", "MSFT", "CAT", "HON", "DE"]

TEST_START = "2022-01-01"   # start of historical evaluation
TEST_END   = None           # e.g. "2024-12-31" or None for last available

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "combined_data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pt"
PRED_OUT_PATH = Path(__file__).resolve().parent.parent / "tcn_historical_predictions.csv"
SUMMARY_OUT_PATH = Path(__file__).resolve().parent.parent / "tcn_backtest_summary.csv"


def classify_decision(ret: float) -> str:
    """
    Convert a return (predicted or realized) into a discrete decision label.
    """
    if ret > BUY_THRESHOLD:
        return "BUY"
    elif ret < HOLD_THRESHOLD:
        return "AVOID"
    else:
        return "HOLD"


def load_model(n_features: int) -> TCN:
    model = TCN(n_features=n_features)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_symbol_df(full_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = full_df[full_df["symbol"] == symbol].copy()
    if df.empty:
        raise RuntimeError(f"No data for symbol {symbol} in combined_data.csv")

    # ensure datetime and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ensure indicators exist
    needed = set(FEATURE_COLS)
    if not needed.issubset(df.columns):
        df = add_indicators(df)

    df = df.dropna().reset_index(drop=True)

    # filter by test period
    if TEST_START is not None:
        df = df[df["date"] >= TEST_START]
    if TEST_END is not None:
        df = df[df["date"] <= TEST_END]

    df = df.reset_index(drop=True)
    return df


def make_walk_forward_predictions(df: pd.DataFrame, model: TCN, symbol: str) -> pd.DataFrame:
    """
    For each time t in the test period, use the previous LOOKBACK days
    to predict the HORIZON-day forward return, and label both prediction and reality.
    """
    feats = df[FEATURE_COLS].values.astype("float32")
    closes = df["close"].values.astype("float32")

    rows = []
    for t in range(LOOKBACK, len(df) - HORIZON):
        window = feats[t - LOOKBACK:t]             # (LOOKBACK, F)
        x = torch.tensor(window).unsqueeze(0)      # (1, LOOKBACK, F)

        with torch.no_grad():
            pred_ret = float(model(x).item())      # predicted HORIZON-day return

        # realized HORIZON-day forward return from t to t+HORIZON
        price_now = closes[t]
        price_future = closes[t + HORIZON]
        real_ret = (price_future / price_now) - 1.0

        pred_label = classify_decision(pred_ret)
        real_label = classify_decision(real_ret)
        correct = int(pred_label == real_label)

        rows.append({
            "symbol": symbol,
            "date": df["date"].iloc[t],
            "pred_return": pred_ret,
            "returns": real_ret,        # actual forward return
            "pred_label": pred_label,   # BUY / HOLD / AVOID (prediction)
            "real_label": real_label,   # BUY / HOLD / AVOID (reality)
            "correct": correct,
        })

    return pd.DataFrame(rows)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python -m data.fetchdata` first.")

    full_df = pd.read_csv(DATA_PATH)
    if full_df.empty:
        raise RuntimeError("combined_data.csv is empty.")

    # infer n_features from one symbol
    sample_sym = SYMBOLS[0]
    tmp = prepare_symbol_df(full_df, sample_sym)
    n_features = len(FEATURE_COLS)

    model = load_model(n_features)

    all_preds = []
    summary_rows = []

    for sym in SYMBOLS:
        try:
            df_sym = prepare_symbol_df(full_df, sym)
        except Exception as e:
            print(f"[{sym}] skipping: {e}")
            continue

        if len(df_sym) < LOOKBACK + HORIZON + 5:
            print(f"[{sym}] not enough rows after filtering (have {len(df_sym)}). Skipping.")
            continue

        print(f"\n>>> Making walk-forward predictions for {sym}")
        preds_df = make_walk_forward_predictions(df_sym, model, sym)
        print(f"    generated {len(preds_df)} prediction steps.")

        if preds_df.empty:
            continue

        # directional accuracy on raw returns sign
        dir_acc = (np.sign(preds_df["pred_return"]) == np.sign(preds_df["returns"])).mean()

        # label accuracy (BUY/HOLD/AVOID match)
        label_acc = preds_df["correct"].mean()

        # use your simple_backtest (long/short based on pred_return sign)
        bt_stats = simple_backtest(preds_df, pred_col="pred_return", cost=0.001)

        # per-label stats (precision-like per predicted label)
        per_label_stats = {}
        for label in ["BUY", "HOLD", "AVOID"]:
            mask = preds_df["pred_label"] == label
            if mask.sum() == 0:
                per_label_stats[f"{label}_n"] = 0
                per_label_stats[f"{label}_acc"] = np.nan
            else:
                acc_label = (preds_df.loc[mask, "pred_label"] == preds_df.loc[mask, "real_label"]).mean()
                per_label_stats[f"{label}_n"] = int(mask.sum())
                per_label_stats[f"{label}_acc"] = acc_label

        summary_rows.append({
            "symbol": sym,
            "steps": len(preds_df),
            "directional_acc": dir_acc,
            "label_acc": label_acc,
            "CAGR": bt_stats["CAGR"],
            "Sharpe": bt_stats["Sharpe"],
            "MaxDrawdown": bt_stats["MaxDrawdown"],
            **per_label_stats,
        })

        all_preds.append(preds_df)

    if not all_preds:
        print("❌ No predictions generated for any symbol.")
        return

    # save all prediction points
    all_df = pd.concat(all_preds, ignore_index=True)
    all_df.to_csv(PRED_OUT_PATH, index=False)
    print(f"\n📁 Saved all historical predictions to: {PRED_OUT_PATH}")

    # build summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n===== TCN Historical Backtest Summary =====")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\n📁 Summary saved to: {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
