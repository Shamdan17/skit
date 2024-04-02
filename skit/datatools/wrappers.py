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
from zipfile import BadZipFile
from collections.abc import Iterable
from skit.distributed import (
    is_dist_avail_and_initialized,
    get_world_size,
    only_on_master,
    save_on_master,
    barrier,
    get_rank,
)
from skit.utils import get_dtype_min_value
import torch.distributed as dist
import time


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
        print("Loading cache from", self.cache_path)

        self._load_cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._iscached(idx):
            try:
                return self._read_from_cache(idx)
            except (BadZipFile, ValueError, EOFError) as e:
                error_type = type(e).__name__
                print(
                    f"Cache file at {self._get_idx_path(idx)} is corrupted and raised a {error_type} error. Deleting and reloading."
                )
                os.remove(self._get_idx_path(idx))
                return self.__getitem__(idx)
        else:
            el = self.dataset[idx]
            save_path = self._get_idx_path(idx)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tmp_save_path = self._get_idx_tmp_path(idx)
            if self.compress:
                np.savez_compressed(
                    tmp_save_path,
                    **self._wrap_data(el),
                )
            else:
                np.savez(
                    tmp_save_path,
                    **self._wrap_data(el),
                )
            os.rename(tmp_save_path, save_path)
            return el

    def _read_from_cache(self, idx):
        return self._unwrap_data(
            {
                k: torch.from_numpy(v)
                # if np.issubdtype(v.dtype, np.number)
                # else (str(v) if np.issubdtype(v.dtype, np.str_) else v)
                for k, v in np.load(self._get_idx_path(idx)).items()
            }
        )

    def _get_idx_path(self, idx):
        if self.block_size > 0:
            block_idx = f"{idx // self.block_size}"
        else:
            block_idx = ""
        return os.path.join(self.cache_path, block_idx, f"{idx:0>6}.npz")

    def _get_idx_tmp_path(self, idx):
        if self.block_size > 0:
            block_idx = f"{idx // self.block_size}"
        else:
            block_idx = ""
        # Get current time
        timestamp = str(time.time())[-4:]
        return os.path.join(
            self.cache_path, block_idx, f"tmp.{timestamp}.{idx:0>6}.npz"
        )

    def _iscached(self, idx):
        return os.path.exists(self._get_idx_path(idx))

    def _clear_tmp_files(self):
        if self.block_size > 0:
            blocks = os.listdir(self.cache_path)
        else:
            blocks = [""]
        for block in blocks:
            if not os.path.isdir(os.path.join(self.cache_path, block)):
                continue
            for f in os.listdir(os.path.join(self.cache_path, block)):
                if f.startswith("tmp"):
                    os.remove(os.path.join(self.cache_path, block, f))

    def _load_cache(self):
        dataset_len = len(self.dataset)
        with only_on_master() as is_main:
            if is_main:
                if not os.path.exists(self.cache_path):
                    os.makedirs(self.cache_path)
                else:
                    self._clear_tmp_files()
        if self.block_size > 0:
            cache_len = sum(
                len(os.listdir(os.path.join(self.cache_path, x)))
                for x in os.listdir(self.cache_path)
                if os.path.isdir(os.path.join(self.cache_path, x))
            )
        else:
            cache_len = len(os.listdir(self.cache_path))

        with only_on_master() as is_main:
            if not is_main:
                return

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
                return
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
                        if isinstance(data[k], np.ndarray):
                            if not (k in cached and np.allclose(data[k], cached[k])):
                                print(
                                    "Cache mismatch at key {}, expected: {}, found: {} overwriting cache".format(
                                        k, data[k], cached[k]
                                    )
                                )
                                torch.save(
                                    {
                                        "expected": data[k],
                                        "found": cached[k],
                                    },
                                    os.path.join(
                                        self.cache_path, f"cache_mismatch_{k}_{i}.pt"
                                    ),
                                )
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
                        print(
                            "Cache not deleted, exiting. Please delete cache manually.",
                            "\nCache path:",
                            self.cache_path,
                        )
                        exit(1)

    def save_state(self):
        pass

    def load_state(self):
        pass

    def infer_dataset_type(self):
        instance = self.dataset[0]
        self.expected_len = len(instance)
        if isinstance(instance, dict):
            print(instance.keys(), self.expected_len)
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
        if len(data) != self.expected_len:
            if self.dtype == "dict":
                print("Invalid len", data.keys(), self.expected_len)
            else:
                print("Invalid len", len(data), self.expected_len)
            raise ValueError
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
        kwargs: Additional arguments to pass to DatasetPreloader
    """

    def __init__(self, dataset, cache_path, **kwargs):
        self.dataset = DatasetPreloader(dataset, cache_path, **kwargs)
        self.cache_path = cache_path
        print("Initializing with cache path", self.cache_path)
        self.cached_count = 0
        self._init_cache()

    def __getitem__(self, idx):
        if self._iscached(idx):
            return self._read_from_cache(idx)

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
            save_on_master(
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


class AugmentableDatasetPreloader(torch.utils.data.Dataset):
    """
    A wrapper around a torch.utils.data.Dataset which caches the dataset to disk. This is
    either done on instantiation or lazily when a batch is requested. Furthermore, this
    wrapper allows the dataset to be augmented on the fly by appending features to the
    original dataset. This is useful to cache expensive preprocessing or feature extraction
    steps.

    Warning: Currently only supports datasets which return a dict.
    > dataset = my_dataset()
    > dataset = AugmentableDatasetPreloader(dataset, cache_path=cache_path)
    > a = dataset[0]
    > output = model(a)
    > features_to_cache = {"feature_1": output["feature_1"], "feature_2": output["feature_2"]}
    > dataset.append_features(a, features)
    > Now, dataset[0] will return a dict with the original features, as well as the appended features.

    # It is highly recommended to do a full pass through the dataset to cache the features first.

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
        augmented_cache_filename=None,
        use_fp16=True,
        **kwargs,
    ):
        self.wrapped_dataset = DatasetPreloader(
            dataset,
            cache_path=cache_path,
            **kwargs,
        )

        self.use_fp16 = use_fp16
        self.dataset = dataset
        self.cache_path = cache_path
        self.append_in_memory = True
        print("Initializing cache", self.cache_path, augmented_cache_filename)
        self.augmented_cache_filename = os.path.join(
            self.cache_path, augmented_cache_filename
        )

        self._init_cache()

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        dct = self.wrapped_dataset[idx]
        dct["_idx"] = idx
        self.load_appended_features(dct)

        return dct

    def _init_cache(self):
        self.initialized = False
        self.appended_features = {}
        self.appended = torch.zeros(len(self.wrapped_dataset), dtype=torch.bool)
        self.appended.share_memory_()

    def load_appended_features(self, dct):
        if self._is_fully_cached():
            idx = dct["_idx"]
            self._read_from_cache(idx, dct)

    def _is_fully_cached(self):
        # print("Checking if fully cached")
        # print(self.appended.sum(), len(self.wrapped_dataset))
        return self.appended.sum() == len(self.wrapped_dataset)

    def _read_from_cache(self, idx, dct):
        if self.append_in_memory:
            for k in self.appended_features.keys():
                dct[k] = self.appended_features[k][idx]
        else:
            raise NotImplementedError("Caching appended to disk not implemented yet.")

    def append_features(self, dct, features):
        # If idx is not iterable, make it iterable by wrapping it in a list
        idxes = dct["_idx"]

        if not isinstance(idxes, Iterable):
            idxes = [idxes]
        else:
            idxes = idxes.flatten().tolist()

        features = {k: v.cpu() for k, v in features.items() if not k.startswith("_")}

        if not self.initialized:
            for k in features.keys():
                dtype = features[k].dtype
                if dtype in [torch.float32, torch.float64] and self.use_fp16:
                    print("Converting", k, "to fp16")
                    dtype = torch.float16

                # Print the size of the array we will allocate and how much memory it will take
                print(
                    "Allocating",
                    k,
                    "with shape",
                    len(self.wrapped_dataset),
                    *(features[k].shape[1:]),
                    "and dtype",
                    dtype,
                    "which will take",
                    len(self.wrapped_dataset)
                    * np.prod(features[k].shape[1:])
                    * features[k].element_size()
                    / 1024**2,
                    "MB",
                )

                self.appended_features[k] = torch.zeros(
                    len(self.wrapped_dataset),
                    *(features[k].shape[1:]),
                    dtype=dtype,
                )
                self.appended_features[k].share_memory_()
            self.initialized = True

        for f_idx, idx in enumerate(idxes):
            if self.appended[idx]:
                continue
            for k in features.keys():
                self.appended_features[k][idx] = features[k][f_idx].to(
                    self.appended_features[k].dtype
                )

            self.appended[idx] = True

    def save_state(self):
        self.wrapped_dataset.save_state()
        self.sync_state()
        if self.append_in_memory:
            save_on_master(
                {
                    "appended_features": self.appended_features,
                    "appended": self.appended,
                },
                self.augmented_cache_filename,
            )
        else:
            raise NotImplementedError("Caching appended to disk not implemented yet.")

    def load_state(self):
        self.wrapped_dataset.load_state()
        if self.append_in_memory:
            if os.path.exists(self.augmented_cache_filename):
                print("Loading augmented cache from", self.augmented_cache_filename)
                state = torch.load(self.augmented_cache_filename)
                print(
                    "State loaded. The state is using {} MB of memory.".format(
                        sum(
                            [
                                v.element_size() * v.nelement()
                                for v in state["appended_features"].values()
                            ]
                        )
                        / 1024**2
                    )
                )
                self.appended_features = state["appended_features"]
                self.appended = state["appended"]
                del state
                if self.use_fp16:
                    # Check if any appended features are float32, if so, convert to fp16
                    must_save = False
                    for k in self.appended_features.keys():
                        if self.appended_features[k].dtype in [
                            torch.float32,
                            torch.float64,
                        ]:
                            print("Converting", k, "to fp16")
                            self.appended_features[k] = self.appended_features[k].half()
                            must_save = True

                    if must_save:
                        save_on_master(
                            {
                                "appended_features": self.appended_features,
                                "appended": self.appended,
                            },
                            self.augmented_cache_filename,
                        )

                print(
                    "Number augmented features cached: {}/{}".format(
                        self.appended.sum(), len(self.appended)
                    )
                )
                for k in self.appended_features.keys():
                    self.appended_features[k].share_memory_()
                self.appended.share_memory_()

    def sync_state(self):
        # Called if there are multiple processes in order to sync the state across processes
        if not is_dist_avail_and_initialized():
            print("Not in distributed mode, skipping sync.")
            return

        assert (
            torch.distributed.is_gloo_available()
        ), "Gloo backend is required for syncing in this version. Please initialize the process group with gloo backend in addition to the NCCL backend"

        full_cached = self._is_fully_cached()

        fully_cached_tensor = torch.tensor([full_cached], dtype=torch.int)

        dist.all_reduce(fully_cached_tensor, op=dist.ReduceOp.PRODUCT)

        if fully_cached_tensor.item():
            print("Dataset is fully cached, skipping sync.")
            return

        world_size = get_world_size()

        appended_array = self.appended.clone().to(torch.int)

        all_appended_arrays = [
            torch.zeros_like(appended_array) for _ in range(world_size)
        ]

        dist.all_gather(all_appended_arrays, appended_array, async_op=False)

        all_appended_arrays = torch.stack(
            all_appended_arrays
        )  # world_size x len(self.wrapped_dataset)

        new_appended_array = all_appended_arrays.sum(dim=0) > 0

        new_appended_where = torch.argmax(all_appended_arrays, dim=0)  #

        new_appended_in_this_process = new_appended_where == get_rank()

        new_data_appended = new_appended_array & ~appended_array

        new_data_found = torch.tensor([(new_data_appended).sum() > 0], dtype=torch.int)

        dist.all_reduce(new_data_found, op=dist.ReduceOp.MAX)

        if not new_data_found.item():
            print("No new data found, skipping sync.")
            return

        # TODO: only sync indices of interest, target for version 0.3.0
        # new_data_appended = new_data_appended.to(torch.int)

        # new_data_appended = dist.all_reduce(new_data_appended, op=dist.ReduceOp.MAX)

        # indices_of_interest = torch.nonzero(new_data_appended).flatten()
        # For now, just sync everything
        # Print current cuda used memory in GB
        keys = list(self.appended_features.keys())
        for k in keys:
            appended_feature_cda = self.appended_features[k].cuda()

            # Make all features not in this process the minimum value possible in order to use max reduction
            appended_feature_cda[~new_appended_in_this_process] = get_dtype_min_value(
                appended_feature_cda.dtype
            )
            dist.all_reduce(appended_feature_cda, op=dist.ReduceOp.MAX)

            del self.appended_features[k]

            # Assert that none of the features are the minimum value
            assert appended_feature_cda.min() != get_dtype_min_value(
                appended_feature_cda.dtype
            ), "Critical error, some features are still undefined after sync."

            self.appended_features[k] = appended_feature_cda.cpu()
            self.appended_features[k].share_memory_()
            del appended_feature_cda

        self.appended = new_appended_array.cpu()
        self.appended.share_memory_()

        print("Syncing finished, total appended:", self.appended.sum().item())

    def __getattr__(self, name):
        return getattr(self.wrapped_dataset, name)
