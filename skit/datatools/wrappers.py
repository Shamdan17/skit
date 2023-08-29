# A wrapper around a dataset
# First, it loads batches, 1 instance at a time, then writes it into a disk cache
# Future calls to load batches will load from the cache instead of the original dataset
# This is to avoid the overhead of loading from disk every time, especially if a single
# instance is loaded from multiple files. More importantly, this allows us skip possibly
# expensive preprocessing steps by only doing it once.

import os
import shutil
import numpy as np
import torch
from tqdm import tqdm


class DatasetPreloader(torch.utils.data.Dataset):
    """
    A wrapper around a torch.utils.data.Dataset which caches the dataset to disk. This is
    either done on instantiation or lazily when a batch is requested. This is to avoid
    the overhead of loading from disk every time, especially if a single instance is
    loaded from multiple files. More importantly, this allows us skip possibly expensive
    preprocessing steps by only doing it once.

    Warning: Currently only supports datasets which return a dict or a tuple of tensors.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to wrap
        cache_path (str): The path to the cache directory
        wipe_cache (bool): Whether to wipe the cache if it exists. Otherwise, if a cache exists
            and is valid, it will be used. If the cache is invalid, the program exits. Default: False
        lazy_loading (bool): Whether to load the entire dataset into memory on
            instantiation or lazily when a batch is requested. Default: True
        compress (bool): Whether to compress the cache. This can save a lot of disk space.
            However, it can be slower to load. It is advised to only turn this off if speed
            is noticeably improved. Default: True
        block_size (int): The number of samples to store in a single folder. This is to avoid
            having too many files in a single directory, which can cause performance issues.
            Set to 0 to disable. Default: 2000
        preloading_workers (int): The number of workers to use when preloading the dataset. Default: 10
        samples_to_confirm_cache (int): The number of samples to check when confirming the cache. Default: 100
    """

    def __init__(
        self,
        dataset,
        cache_path,
        wipe_cache=False,
        lazy_loading=True,
        compress=True,
        block_size=2000,
        preloading_workers=10,
        samples_to_confirm_cache=100,
    ):
        self.dataset = dataset
        self.cache_path = cache_path
        self.pre_load = not lazy_loading
        self.block_size = block_size
        self.compress = compress
        self.wipe_cache = wipe_cache

        self.preloading_workers = preloading_workers

        self.samples_to_confirm_cache = samples_to_confirm_cache

        self.infer_dataset_type()

        self._load_cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._iscached(idx):
            return self._read_from_cache(idx)
        else:
            el = self.dataset[idx]
            save_path = self._get_idx_path(idx)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if self.compress:
                np.savez_compressed(
                    self._get_idx_path(idx),
                    **self._wrap_data(el),
                )
            else:
                np.savez(
                    self._get_idx_path(idx),
                    **self._wrap_data(el),
                )
            return el

    def _read_from_cache(self, idx):
        return self._unwrap_data(
            {
                k: torch.from_numpy(v)
                for k, v in np.load(self._get_idx_path(idx)).items()
            }
        )

    def _get_idx_path(self, idx):
        if self.block_size > 0:
            block_idx = f"{idx // self.block_size}"
        else:
            block_idx = ""
        return os.path.join(self.cache_path, block_idx, f"{idx:0>6}.npz")

    def _iscached(self, idx):
        return os.path.exists(self._get_idx_path(idx))

    def _load_cache(self):
        dataset_len = len(self.dataset)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if self.block_size > 0:
            cache_len = sum(
                len(os.listdir(os.path.join(self.cache_path, x)))
                for x in os.listdir(self.cache_path)
                if os.path.isdir(os.path.join(self.cache_path, x))
            )
        else:
            cache_len = len(os.listdir(self.cache_path))

        if cache_len != dataset_len:
            print("Cache not found, creating cache")

            # Use a dataloader to have multiple workers
            if self.pre_load:
                loader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=None,
                    num_workers=self.preloading_workers,
                    shuffle=False,
                )

                for idx, data in enumerate(tqdm(loader, desc="Preloading Data")):
                    if self.compress:
                        np.savez_compressed(
                            os.path.join(self.cache_path, f"{idx:0>6}.npz"),
                            **self._wrap_data(data),
                        )
                    else:
                        np.savez(
                            os.path.join(self.cache_path, f"{idx:0>6}.npz"),
                            **self._wrap_data(data),
                        )
        else:
            # Randomly sample samples_to_confirm_cache elements from the dataset, check if they match the cache
            els = np.random.choice(
                dataset_len,
                min(dataset_len, self.samples_to_confirm_cache),
                replace=False,
            )

            invalid_cache = False

            for i in els:
                if not self._iscached(i):
                    continue
                cached = self._read_from_cache(i)
                data = self.dataset[i]

                for k in data.keys():
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].numpy()

                for k in data.keys():
                    if isinstance(data[k], np.ndarray):
                        if not np.allclose(data[k], cached[k]):
                            print("Cache mismatch, overwriting cache")
                            invalid_cache = True
                            break

                if invalid_cache:
                    break

            if invalid_cache or self.wipe_cache:
                # Ask user for manual permission by entering Y before proceeding
                # Even if the user enters Y, the cache will not be deleted if the cache path is not a subdirectory of the current working directory
                confirmation = input(
                    f"Confirm deletion of cache dir at: {self.cache_path}. [y/N]"
                )

                if (
                    os.path.abspath(self.cache_path).startswith(
                        os.path.abspath(os.getcwd())
                    )
                    and confirmation.lower() == "y"
                ):
                    shutil.rmtree(self.cache_path)
                else:
                    print("Cache not deleted, exiting. Please delete cache manually.")
                    exit(1)

    def save_state(self):
        pass

    def load_state(self):
        pass

    def infer_dataset_type(self):
        instance = self.dataset[0]

        if isinstance(instance, dict):
            self.dtype = "dict"
        elif isinstance(instance, tuple):
            self.dtype = "tuple"
        else:
            raise ValueError(
                "Dataset must return a dict or tuple. Other types not supported.\n"
                "If you would like to use a custom dataset, please implement the logic or open an issue on the repo.\n"
                "Please include a minimal reproducible example."
            )

    def _unwrap_data(self, data):
        if self.dtype == "dict":
            return {k: v for k, v in data.items()}
        elif self.dtype == "tuple":
            return tuple(data[f"array_{i}"] for i in range(len(data)))
        else:
            raise ValueError(
                "Dataset must return a dict or tuple. Other types not supported.\n"
                "If you would like to use a custom dataset, please implement the logic or open an issue on the repo.\n"
                "Please include a minimal reproducible example."
            )

    def _wrap_data(self, data):
        if self.dtype == "dict":
            return data
        elif self.dtype == "tuple":
            return {f"array_{i}": v for i, v in enumerate(data)}
        else:
            raise ValueError(
                "Dataset must return a dict or tuple. Other types not supported.\n"
                "If you would like to use a custom dataset, please implement the logic or open an issue on the repo.\n"
                "Please include a minimal reproducible example."
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class InMemoryDatasetPreloader(torch.utils.data.Dataset):
    """
    A wrapper around a torch.utils.data.Dataset which caches the dataset to memory. This is to avoid
    the overhead of loading from disk every time, and allows us skip possibly expensive
    preprocessing steps by only doing it once. May not be suitable for large datasets. Internally,
    the given dataset is wrapped in a DatasetPreloader.

    Warning: Currently only supports datasets which return a dict or a tuple of tensors.
    Args:
        dataset (torch.utils.data.Dataset): The dataset to wrap
        cache_path (str): The path to the cache directory
        wipe_cache (bool): Whether to wipe the cache if it exists. Otherwise, if a cache exists
            and is valid, it will be used. If the cache is invalid, the program exits. Default: False
        kwargs: Additional arguments to pass to DatasetPreloader
    """

    def __init__(self, dataset, cache_path, wipe_cache=False, **kwargs):
        self.dataset = DatasetPreloader(dataset, cache_path, wipe_cache, **kwargs)
        self.cache_path = cache_path
        self.cached_count = 0
        self._init_cache()

    def __getitem__(self, idx):
        if self._iscached(idx):
            return self._read_from_cache(idx)
        else:
            el = self.dataset[idx]
            el = torch.utils.data._utils.collate.default_convert(el)
            el_wrapped = self.dataset._wrap_data(el)
            for k in el_wrapped.keys():
                self.cache[k][idx] = el_wrapped[k]
            self.cached[idx] = True

            return el

    def __len__(self):
        return len(self.dataset)

    def _read_from_cache(self, idx):
        return self.dataset._unwrap_data(
            {k: self.cache[k][idx] for k in self.cache.keys()}
        )

    def _iscached(self, idx):
        return self.cached[idx]

    def _init_cache(self):
        idx = 0
        sample = self.dataset[idx]
        # Collate to convert to tensors
        sample = torch.utils.data._utils.collate.default_convert(sample)
        sample = self.dataset._wrap_data(sample)

        d_len = len(self.dataset)

        self.cache = {}
        self.cached = torch.zeros(d_len, dtype=torch.bool)
        self.cached.share_memory_()
        for k in sample.keys():
            self.cache[k] = torch.zeros(
                (d_len, *sample[k].shape), dtype=sample[k].dtype
            )
            self.cache[k].share_memory_()

            self.cache[k][idx] = sample[k]

        self.cached[idx] = True

    def is_entirely_cached(self):
        return self.cached.sum() == len(self.dataset)

    def save_state(self):
        cached_count = self.cached.sum()
        if cached_count > self.cached_count:
            print("Cache update detected, saving updated state.")
            self.cached_count = cached_count
            torch.save(
                {"cached": self.cached, "cache": self.cache},
                os.path.join(self.cache_path, "wrapper_state.pt"),
            )

    def load_state(self):
        cache_path = os.path.join(self.cache_path, "wrapper_state.pt")
        if os.path.exists(cache_path):
            print("Loading data state")
            state = torch.load(cache_path)

            self.cached = state["cached"]
            self.cache = state["cache"]

            self.cache.share_memory_()
            self.cached.share_memory_()
            self.cached_count = self.cached.sum()

    def __getattr__(self, name):
        return getattr(self.dataset, name)
