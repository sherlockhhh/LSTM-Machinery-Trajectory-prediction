import torch


def test_prepared_windows_have_42_feature_channels(tiny_hdf5):
    from dataset import prepare_splits

    splits = prepare_splits(tiny_hdf5, history_length=3, prediction_horizon=2, seed=42)
    history, target_delta, last_pos, target_abs = splits.train[0]

    assert history.shape == (3, 42)
    assert target_delta.shape == (2, 3)
    assert last_pos.shape == (3,)
    assert target_abs.shape == (2, 3)
    assert all(isinstance(value, torch.Tensor) for value in (history, target_delta, last_pos, target_abs))


def test_demo_identifiers_do_not_overlap_between_splits(tiny_hdf5):
    from dataset import prepare_splits

    splits = prepare_splits(tiny_hdf5, history_length=3, prediction_horizon=2, seed=42)
    train_ids = set(splits.train_demo_ids)
    valid_ids = set(splits.valid_demo_ids)
    test_ids = set(splits.test_demo_ids)

    assert not train_ids & valid_ids
    assert not train_ids & test_ids
    assert not valid_ids & test_ids
    assert len(train_ids | valid_ids | test_ids) == 10
