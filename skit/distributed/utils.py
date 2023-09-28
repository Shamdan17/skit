import torch
import torch.distributed as dist


def disable_printing(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


class NotMasterException(Exception):
    """Dummy exception to be raised when the current process is not the master process"""


class only_on_master:
    def __init__(self):
        pass

    def __enter__(self):
        if not is_main_process():
            raise NotMasterException()

    def __exit__(self, exc_type, exc_value, traceback):
        barrier()
        if exc_type is NotMasterException:
            return True
        return False  # To re-raise the exception if it's not a NotMasterException
