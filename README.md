# skit: Shadi's Toolkit

A set of useful tools I use throughout my codebases.

To install: 

```
git clone git@github.com:Shamdan17/skit.git
pip install -e .
```

## Dataset Wrappers

### Preloader
DatasetPreloader is a wrapper around `torch.utils.data.Dataset` which caches the dataset to disk. This is either done on instantiation or lazily when a batch is requested. This is to avoid the overhead of loading from disk every time, especially if a single instance is loaded from multiple files. More importantly, this allows us skip possibly expensive preprocessing steps by only doing it once.

Warning: Currently only supports datasets which return a dict or a tuple of tensors.

Usage:
```python
from skit.data import DatasetPreloader

dataset = myTorchDataset()
cache_path = 'path/to/cache'

# Wrap the dataset
dataset = DatasetPreloader(
    dataset,
    cache_path=cache_path,
    wipe_cache=False, # If the cache exists, use it. Otherwise, create it. If true, delete the cache if it exists.
    lazy_loading=True, # Load the entire dataset into memory on instantiation or lazily when a batch is requested
    compress=True, # Compress the cache. This can save a lot of disk space. However, it can be slower to load.
    block_size=2000, # The number of samples to store in a single folder. This is to avoid having too many files in a single directory, which can cause performance issues. Set to 0 to disable.
    preloading_workers=10, # The number of workers to use when preloading the dataset. Does not affect lazy loading.
    samples_to_confirm_cache=100 # The number of samples to check when confirming the cache. If your dataset has many instances, increase the number of samples to confirm the cache. Please note this process is only a heuristic and is not 100% accurate. If in doubt, wipe the cache.
)

# Access the dataset as normal
```

### InMemoryDatasetPreloader
InMemoryDatasetPreloader is a wrapper on top of DatasetPreloader which loads the entire dataset into memory. This is useful if you have a small dataset and want to avoid the overhead of loading from disk every time. Has the exact same API as DatasetPreloader.

