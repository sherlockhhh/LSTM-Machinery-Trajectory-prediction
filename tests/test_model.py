import torch


def test_lstm_returns_one_three_dimensional_path_per_sample():
    from model import LSTMTrajectoryPredictor

    model = LSTMTrajectoryPredictor(
        input_dim=42,
        hidden_dim=16,
        num_layers=2,
        horizon=4,
        dropout=0.2,
    )

    assert model(torch.zeros(5, 3, 42)).shape == (5, 4, 3)
