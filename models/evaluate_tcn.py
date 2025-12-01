from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data.dataset import create_sequences
from data.indicators import add_indicators
from models.tcn_model import TCN

LOOKBACK = 60
HORIZON = 5
SEED = 42

torch.manual_seed(SEED)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "combined_data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pt"

# 👉 Choose your stocks here (IT + Industrials)
SYMBOLS = ["NVDA", "AAPL", "MSFT", "CAT", "HON", "DE"]

df_all = pd.read_csv(DATA_PATH)

# Load trained model
model = TCN(n_features=6)  # depends on number of indicators
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

results = []   # results table for report

for SYMBOL in SYMBOLS:
    print(f"\n============================")
    print(f"Evaluating TCN on {SYMBOL}")
    print(f"============================")

    df = df_all[df_all["symbol"] == SYMBOL].copy()

    # Ensure indicators exist
    needed_cols = {"SMA_10","SMA_50","RSI_14","MACD","MACD_signal","returns"}
    if not needed_cols.issubset(df.columns):
        df = add_indicators(df)

    df = df.dropna().reset_index(drop=True)

    # Build sequences
    X, y = create_sequences(df, lookback=LOOKBACK, target_horizon=HORIZON)
    n = len(X)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    X_test = X[val_end:]
    y_test = y[val_end:]

    with torch.no_grad():
        preds = model(X_test).squeeze().numpy()

    y_test_np = y_test.numpy()

    # Metrics
    mse  = np.mean((preds - y_test_np)**2)
    mae  = np.mean(np.abs(preds - y_test_np))
    acc  = np.mean(np.sign(preds)==np.sign(y_test_np))

    # Backtest
    start_idx = LOOKBACK + val_end
    end_idx = start_idx + len(preds)
    test_df = df.iloc[start_idx:end_idx].copy()
    test_df["pred_return"] = preds
    test_df["signal"] = (test_df["pred_return"] > 0).astype(int)
    test_df["strategy_ret"] = test_df["signal"].shift(1) * test_df["returns"]
    test_df["equity"] = (1 + test_df["strategy_ret"].fillna(0)).cumprod()

    ann_factor = 252
    sharpe = (np.sqrt(ann_factor) *
              test_df["strategy_ret"].mean() /
              (test_df["strategy_ret"].std() + 1e-9))

    cagr = test_df["equity"].iloc[-1] ** (ann_factor/len(test_df)) - 1
    mdd  = (test_df["equity"].cummax() - test_df["equity"]).max() \
            / test_df["equity"].cummax().max()

    print(f"Test MSE:  {mse:.6f}")
    print(f"Test MAE:  {mae:.6f}")
    print(f"Direction Accuracy: {acc*100:.2f}%")
    print(f"CAGR: {cagr*100:.2f}%, Sharpe: {sharpe:.2f}, MaxDD: {mdd*100:.2f}%")

    # Save results for a summary table
    results.append([SYMBOL, mse, mae, acc, cagr, sharpe, mdd])

# ---- Final table for your report ----
results_df = pd.DataFrame(results, columns=[
    "Symbol", "MSE", "MAE", "Direction_Acc",
    "CAGR", "Sharpe", "Max_Drawdown"
])
out = Path(__file__).resolve().parent.parent / "evaluation_results.csv"
results_df.to_csv(out, index=False)
print("\n📁 Summary saved to:", out)
print(results_df)
