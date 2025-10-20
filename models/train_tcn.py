import torch, os
from data.fetchdata import get_stock_data
from data.dataset import create_sequences
from models.tcn_model import TCN

SYMBOLS = ["NVDA", "DTEGY"]
LOOKBACK = 60
EPOCHS = 100
LR = 1e-3

# Gather data
dfs = [get_stock_data(s, "2015-01-01", "2025-01-01") for s in SYMBOLS]
df = torch.cat([torch.tensor(create_sequences(d, LOOKBACK)[0]) for d in dfs])
X, y = create_sequences(dfs[0], LOOKBACK)  # adapt for multiasset later

split = int(len(X)*0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

model = TCN(n_features=X.shape[2])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred.squeeze(), y_train)
    loss.backward(); optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        val_loss = criterion(model(X_val).squeeze(), y_val)
        print(f"{epoch:03d}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
print("✅ model saved.")
