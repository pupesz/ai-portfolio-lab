"""
experiment/models_factory.py
============================

Factory functions that build the two deep models with a matched capacity,
so experiment code can stay model-agnostic (just ask for 'tcn' or 'lstm').
"""
from __future__ import annotations

import torch.nn as nn

from models.tcn_model import TCN
from models.lstm_model import LSTMRegressor

# Shared capacity: TCN channel width == LSTM hidden size.
HIDDEN = 64
DROPOUT = 0.1


def make_tcn(n_features: int) -> nn.Module:
    return TCN(
        n_features=n_features,
        channels=HIDDEN,
        levels=5,
        kernel_size=7,
        dropout=DROPOUT,
        out_dim=1,
        head_norm=True,  # stabilises regression on the zero-centred return target
    )


def make_lstm(n_features: int) -> nn.Module:
    return LSTMRegressor(
        n_features=n_features,
        hidden_size=HIDDEN,
        num_layers=2,
        dropout=DROPOUT,
        out_dim=1,
    )


MODEL_FACTORIES = {"TCN": make_tcn, "LSTM": make_lstm}
