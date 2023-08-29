from skit import DatasetPreloader, InMemoryDatasetPreloader

import torch
import pytest
from torch.utils.data import DataLoader
import numpy as np
from .dummy_datasets import (
    DummyTupleDataset,
    DummyDictDataset,
    DummyIntDataset,
)
from .utils import equals_pytorch
import shutil

preloader_map = {
    "inmemory": InMemoryDatasetPreloader,
    "ondisk": DatasetPreloader,
}
DATASET_MAP = {
    "DummyTupleDataset": DummyTupleDataset,
    "DummyDictDataset": DummyDictDataset,
    # "DummyIntDataset": DummyIntDataset,
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
def test_dataset_preloader_lazy(
    preloader_name, preloader_class, dataset_name, dataset_class
):
    dataset = dataset_class()

    cache_path = f"./.cache/{dataset_name}_{preloader_name}"
    # Make sure cache directory is empty
    shutil.rmtree(cache_path, ignore_errors=True)

    # Load entire dataset into memory first with normal dataloader
    regulardataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    entries = []

    for batch in regulardataloader:
        entries.append(batch)

    wrappeddataset = preloader_class(dataset, cache_path=cache_path)
    preloader = DataLoader(wrappeddataset, batch_size=8, shuffle=False)
    # First pass does not load from cache
    for batch, entry in zip(preloader, entries):
        assert equals_pytorch(
            batch, entry
        ), f"{dataset_name} failed in first pass. Expected {entry}, got {batch}."

    # Second pass loads from cache
    for batch, entry in zip(preloader, entries):
        assert equals_pytorch(
            batch, entry
        ), f"{dataset_name} failed in second pass. Expected {entry}, got {batch}."


@pytest.mark.parametrize(
    "preloader_name,preloader_class,dataset_name,dataset_class",
    all_combinations,
)
def test_dataset_preloader_precaching(
    preloader_name, preloader_class, dataset_name, dataset_class
):
    dataset = dataset_class()

    cache_path = f"./.cache/{dataset_name}_{preloader_name}_precaching"
    # Make sure cache directory is empty
    shutil.rmtree(cache_path, ignore_errors=True)

    # Load entire dataset into memory first with normal dataloader
    regulardataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    entries = []

    for batch in regulardataloader:
        entries.append(batch)

    wrappeddataset = preloader_class(
        dataset,
        cache_path=cache_path,
        lazy_loading=False,
    )
    preloader = DataLoader(wrappeddataset, batch_size=8, shuffle=False)
    # First pass should load from cache
    for batch, entry in zip(preloader, entries):
        assert equals_pytorch(
            batch, entry
        ), f"{dataset_name} failed in first pass after preloading. Expected {entry}, got {batch}."


@pytest.mark.parametrize(
    "dataset_name,dataset_class,dtype",
    [
        ("DummyTupleDataset", DummyTupleDataset, "tuple"),
        ("DummyDictDataset", DummyDictDataset, "dict"),
    ],
)
def test_dataset_preloader_dtype(
    dataset_name,
    dataset_class,
    dtype,
):
    dataset = dataset_class()
    cache_path = f"./.cache/{dataset_name}_{dtype}"
    wrapped = DatasetPreloader(dataset, cache_path=cache_path)

    assert wrapped.dtype == dtype


@pytest.mark.parametrize(
    "wrapper_name,wrapper_class",
    [
        ("inmemory", InMemoryDatasetPreloader),
        ("ondisk", DatasetPreloader),
    ],
)
def test_accessing_dataset_attribute(wrapper_name, wrapper_class):
    dataset = DummyTupleDataset()
    cache_path = f"./.cache/{wrapper_name}_dataset_attribute_test"
    wrapped = wrapper_class(dataset, cache_path=cache_path)

    # Access the data attribute
    wrapped.data

    assert wrapped.dummy_method() == 1
