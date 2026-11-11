"""
models/lstm_model.py
====================

Comparative LSTM regressor for the 5-day forward return target.

Design mirrors models/tcn_model.py so the comparison is fair:
  * same input  : (batch, lookback, n_features)
  * same output : single scalar regression head
  * comparable capacity : 2 stacked LSTM layers whose hidden size matches
    the TCN channel width (default 64), dropout between layers, linear head.
"""
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, out_dim)

    def forward(self, x):  # x: (batch, time, features)
        out, _ = self.lstm(x)         # (batch, time, hidden)
        last = out[:, -1, :]          # last timestep
        return self.head(self.dropout(last))
