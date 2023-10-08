import torch


def get_dtype_min_value(dtype):
    try:
        return torch.iinfo(dtype).min
    except TypeError:
        pass

    try:
        return torch.finfo(dtype).min
    except TypeError:
        pass

    raise ValueError(f"Unsupported dtype {dtype}")
