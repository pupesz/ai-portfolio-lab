import torch
import numpy as np
import pandas as pd
from typing import Tuple

FEATURE_COLS = ["SMA_10", "SMA_50", "RSI_14", "MACD", "MACD_signal", "returns"]

def create_sequences(df: pd.DataFrame, lookback: int = 60, target_horizon: int = 5
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds (X, y) tensors for time series supervised learning.
    X shape: (N, lookback, features)
    y shape: (N,)
    """
    features = df[FEATURE_COLS].values
    target = df["returns"].shift(-target_horizon).values
    X, y = [], []
    for i in range(len(features) - lookback - target_horizon):
        X.append(features[i:i + lookback])
        y.append(target[i + lookback])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y
