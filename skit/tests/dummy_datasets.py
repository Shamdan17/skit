# A bunch of dummy datasets to test the functionality of the library.

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .utils import generate_random_array


class DummyTupleDataset(Dataset):
    def __init__(self, seed=10, size=1000, dim=3):
        self.data = []
        np_rng = np.random.default_rng(seed)

        shapes = np_rng.integers(low=1, high=100, size=dim)
        dtypes = np_rng.choice([np.float32, np.float64, np.int32, np.int64], size=dim)

        for _ in range(size):
            self.data.append(
                tuple(
                    generate_random_array(np_rng, dtype, shape)
                    for dtype, shape in zip(dtypes, shapes)
                )
            )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class DummyDictDataset(Dataset):
    def __init__(self, seed=10, size=1000, dim=3):
        self.data = []
        np_rng = np.random.default_rng(seed)

        shapes = np_rng.integers(low=1, high=100, size=dim)
        dtypes = np_rng.choice([np.float32, np.float64, np.int32, np.int64], size=dim)

        for _ in range(size):
            self.data.append(
                {
                    str(i): generate_random_array(np_rng, dtype, shape)
                    for i, (dtype, shape) in enumerate(zip(dtypes, shapes))
                }
            )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class DummyIntDataset(Dataset):
    def __init__(self, seed=10, size=1000):
        self.data = []
        np_rng = np.random.default_rng(seed)

        for _ in range(size):
            self.data.append(np_rng.integers(low=0, high=100, size=1)[0])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
