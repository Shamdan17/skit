from skit import DatasetPreloader, InMemoryDatasetPreloader, AugmentableDatasetPreloader

import torch
import pytest
from torch.utils.data import DataLoader
import numpy as np
from .dummy_datasets import (
    DummyDictOnesDataset,
)
from .utils import equals_pytorch
import shutil
from copy import deepcopy

preloader_map = {
    "inmemoryaugmentable": AugmentableDatasetPreloader,
}
DATASET_MAP = {
    "DummyDictOnesDataset": DummyDictOnesDataset,
}

from itertools import product

all_combinations = list(product(preloader_map.items(), DATASET_MAP.items()))
all_combinations = [
    (preloader_name, preloader_class, dataset_name, dataset_class)
    for (preloader_name, preloader_class), (
        dataset_name,
        dataset_class,
    ) in all_combinations
]


@pytest.mark.parametrize(
    "preloader_name,preloader_class,dataset_name,dataset_class",
    all_combinations,
)
def test_dataset_preloader_appending(
    preloader_name, preloader_class, dataset_name, dataset_class
):
    dataset = dataset_class()

    cache_path = f"./.cache/{dataset_name}_{preloader_name}"
    # Make sure cache directory is empty
    shutil.rmtree(cache_path, ignore_errors=True)

    # Load entire dataset into memory first with normal dataloader
    regulardataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    entries = []
    augmented_entries = []

    fn = lambda x: 2 * x

    for batch in regulardataloader:
        entries.append(batch)

        # Augment the batch
        augmented_entries.append(deepcopy(batch))

        augmented_entries[-1]["auged"] = fn(augmented_entries[-1]["ones"])

    wrappeddataset = preloader_class(dataset, cache_path=cache_path)
    preloader = DataLoader(wrappeddataset, batch_size=8, shuffle=False)
    # First pass does not load from cache
    for batch, entry in zip(preloader, entries):
        assert equals_pytorch(
            batch, entry
        ), f"{dataset_name} failed in first pass. Expected {entry}, got {batch}."

        auged_dict = {}
        auged_dict["auged"] = fn(entry["ones"])

        wrappeddataset.append_features(batch, auged_dict)

    # Second pass loads augmented batches from cache
    for batch, entry in zip(preloader, augmented_entries):
        assert equals_pytorch(
            batch, entry
        ), f"{dataset_name} failed in second pass. Expected {entry}, got {batch}."
