import numpy as np
import torch


def generate_random_array(np_rng, dtype, shape):
    if dtype == np.float32:
        return np_rng.random(shape).astype(np.float32)
    elif dtype == np.float64:
        return np_rng.random(shape).astype(np.float64)
    elif dtype == np.int32:
        return np_rng.integers(low=0, high=100, size=shape).astype(np.int32)
    elif dtype == np.int64:
        return np_rng.integers(low=0, high=100, size=shape).astype(np.int64)
    else:
        raise ValueError("Invalid dtype: {}".format(dtype))


def equals_pytorch(a, b, verbose=True):
    # If both tensors, use torch.equal
    # If both mappings, use equals_pytorch on each value
    # If both sequences, use equals_pytorch on each value
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        result = torch.equal(a, b)
        if verbose and not result:
            print("Tensors not equal:")
            print(a)
            print(b)
        return result
    elif isinstance(a, dict) and isinstance(b, dict):
        if len(a) != len(b):
            if verbose:
                print(
                    "Dicts not equal: different lengths. |a|={}, |b|={}.\n Content: a: {}, b: {}".format(
                        len(a), len(b), a, b
                    )
                )
            return False
        for key in a:
            if key not in b:
                if verbose:
                    print(
                        "Dicts not equal: key {} not in b. |a|={}, |b|={}.\n Content: a: {}, b: {}".format(
                            key, len(a), len(b), a, b
                        )
                    )
                return False
            if not equals_pytorch(a[key], b[key]):
                if verbose:
                    print(
                        "Dicts not equal: key {} not equal. |a|={}, |b|={}.\n Content: a: {}, b: {}".format(
                            key, len(a), len(b), a[key], b[key]
                        )
                    )
                return False
        return True
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            if verbose:
                print(
                    "Sequences not equal: different lengths. |a|={}, |b|={}.\n Content: a: {}, b: {}".format(
                        len(a), len(b), a, b
                    )
                )
            return False
        for i in range(len(a)):
            if not equals_pytorch(a[i], b[i]):
                if verbose:
                    print(
                        "Sequences not equal: index {} not equal. |a|={}, |b|={}.\n Content: a: {}, b: {}".format(
                            i, len(a), len(b), a[i], b[i]
                        )
                    )
                return False
        return True
    # Otherwise use direct comparison
    else:
        result = a == b
        if verbose and not result:
            print("Not equal:")
            print(a)
            print(b)
        return result
    # else:
    #     raise ValueError(
    #         "Invalid types for equals_pytorch: {}, {}".format(type(a), type(b))
    #     )
