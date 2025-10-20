# models/train_tcn.py
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import create_sequences
from data.indicators import add_indicators
from data.fetchdata import build_universe  # fallback if CSV missing
from models.tcn_model import TCN

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
SYMBOL = "NVDA"
LOOKBACK = 60
HORIZON = 5
EPOCHS = 50
LR = 1e-3
BATCH = 128
SEED = 42

torch.manual_seed(SEED)

# Path to combined dataset
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "combined_data.csv"

# --------------------------------------------------------------------
# Load or build dataset
# --------------------------------------------------------------------
if DATA_PATH.exists():
    print(f"✅ Loading local dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    print("⚠️ No combined dataset found — downloading fresh data...")
    df = build_universe()

# Keep only one symbol if training single-stock model
if "symbol" in df.columns:
    df = df[df["symbol"] == SYMBOL].copy()

# Ensure indicators exist (if you loaded a raw dataset)
if not {"SMA_10", "RSI_14", "MACD"}.issubset(df.columns):
    print("ℹ️  Adding indicators to dataset...")
    df = add_indicators(df)

df = df.dropna().reset_index(drop=True)
print(f"📊 Training data shape: {df.shape}")

# --------------------------------------------------------------------
# Build sequences for deep model
# --------------------------------------------------------------------
X, y = create_sequences(df, lookback=LOOKBACK, target_horizon=HORIZON)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]
print(f"📈 Train samples: {len(X_train)}, Val samples: {len(X_val)}")

# --------------------------------------------------------------------
# Initialize model
# --------------------------------------------------------------------
model = TCN(n_features=X.shape[2])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()


def batch_iter(X, y, bs):
    for i in range(0, len(X), bs):
        yield X[i:i + bs], y[i:i + bs]


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for xb, yb in batch_iter(X_train, y_train, BATCH):
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).squeeze()
        val_loss = criterion(val_pred, y_val).item()

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), Path("model.pt"))

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

print("✅ Done — best model saved as model.pt")
