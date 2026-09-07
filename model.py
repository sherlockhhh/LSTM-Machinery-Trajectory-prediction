"""LSTM encoder with a multi-step Cartesian trajectory regression head."""

from __future__ import annotations

import torch
from torch import nn


class LSTMTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, num_layers, horizon) < 1:
            raise ValueError("input_dim, hidden_dim, num_layers and horizon must be positive.")
        self.prediction_horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # LSTM's built-in dropout is disabled for one layer. Applying it
            # in the regression head regularizes both one- and multi-layer models.
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * 3),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, history_length, input_dim].")
        _, (hidden_state, _) = self.lstm(history)
        output = self.head(hidden_state[-1])
        return output.view(history.shape[0], self.prediction_horizon, 3)
