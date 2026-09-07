import h5py
import numpy as np
import pytest


@pytest.fixture
def tiny_hdf5(tmp_path):
    path = tmp_path / "tiny_robomimic.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_index in range(10):
            demo = data.create_group(f"demo_{demo_index}")
            obs = demo.create_group("obs")
            steps = 8
            base = np.arange(steps, dtype=np.float32)[:, None] + demo_index
            obs.create_dataset("robot0_eef_pos", data=np.concatenate([base, base + 1, base + 2], axis=1))
            obs.create_dataset("robot0_eef_quat", data=np.tile([0.0, 0.0, 0.0, 1.0], (steps, 1)))
            obs.create_dataset("robot0_joint_pos", data=np.tile(base, (1, 7)))
            obs.create_dataset("robot0_joint_vel", data=np.tile(base * 0.1, (1, 7)))
            obs.create_dataset("object", data=np.tile(base, (1, 14)))
            demo.create_dataset("actions", data=np.tile(base * 0.01, (1, 7)))
    return str(path)
