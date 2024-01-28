import torch
import torch.distributed as dist
import numpy as np


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


class only_on_master:
    def __init__(self):
        pass

    def __enter__(self):
        return is_main_process()

    def __exit__(self, exc_type, exc_value, traceback):
        barrier()
        return False  # To re-raise the exception if it's not a NotMasterException


def check_sampler_index_consistency(
    indices,
    summary_only=True,
    verbose=False,
):
    """
    This function checks if the indices across all processes are mutually exclusive
    """
    num_indices = len(indices)
    max_len_indices = torch.tensor([num_indices], dtype=torch.int64).cuda()

    # Get the max length of indices across all processes
    dist.all_reduce(max_len_indices, op=dist.ReduceOp.MAX)

    # Pad the indices to the max length with -1
    indices = torch.tensor(indices, dtype=torch.int64).cuda()
    indices = torch.cat(
        [
            indices,
            torch.ones(max_len_indices - num_indices, dtype=torch.int64).cuda() * -1,
        ],
        dim=0,
    )

    world_size = get_world_size()

    all_indices = [torch.zeros_like(indices) for _ in range(world_size)]
    dist.all_gather(all_indices, indices)

    all_indices = [x.cpu().numpy() for x in all_indices]

    consistent = True

    for i in range(1, world_size):
        # Check if intersection is empty other than possibly -1
        intersection = np.intersect1d(all_indices[0], all_indices[i])
        if len(intersection)>1 and intersection[0] == -1:
            intersection = intersection[1:]

        if len(intersection) > 0:
            if verbose:
                print("Rank {} and {} has overlapping indices".format(0, i))
                print("Intersection: {}".format(intersection))
            consistent = False

    if summary_only and is_main_process() and not consistent:
        print("Found overlapping indices across processes.")

    barrier()
    return consistent


def check_parameter_consistency(
    model,
    summary_only=True,
    return_diffs=False,
    verbose=False,
    summary_rows=10,
    skip_if_grad_found=True,
):
    """
    This function checks if all the parameters across all processes are the same
    Only do this on main process
    """
    if not is_dist_avail_and_initialized():
        return True

    consistent = True

    world_size = get_world_size()

    # To store the l2 norm of the differences
    l2_diffs = []

    for name, param in model.named_parameters():
        if skip_if_grad_found and param.grad is not None:
            barrier()
            return True
        tensor = param.data
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        if not is_main_process():
            # We only check in main process
            # We need the all_gather to be done in all processes because it contains a barrier
            # Could be fixed easily probably, but I didn't bother looking too much into it.
            continue
        for i in range(1, world_size):
            if not torch.equal(tensor_list[0], tensor_list[i]):
                if verbose:
                    print(
                        "Rank {} and {} has inconsistent parameter {}".format(
                            0, i, name
                        )
                    )

                # Get l2 norm of the difference
                diff = tensor_list[0] - tensor_list[i]
                l2_norm = torch.norm(diff)
                if verbose and not summary_only:
                    print("l2 norm of difference: {:.4e}".format(l2_norm))
                l2_diffs.append((l2_norm, name, 0, i))
                consistent = False

    widths = [25, 16, 25]

    if summary_only and is_main_process() and not consistent:
        print("Found inconsistent parameters across processes. ")
        l2_diffs.sort(key=lambda x: x[0], reverse=True)
        min_param_width = max([len(x[1]) for x in l2_diffs[:summary_rows]])
        widths[0] = max(widths[0], min_param_width)
        table_width = sum(widths) + 4
        # Pretty print as follows:
        # ----------------------------------------------------------------
        # |     Parameter     | Ranks compared |  L2 norm of difference  |
        # Widths are 19, 16, and 25 characters respectively
        # We center the text in the middle of the width
        # Also, sort by l2 norm of difference, descending. Only print summary_rows
        print("-" * table_width)
        print(
            "|{}|{}|{}|".format(
                "Parameter".center(widths[0]),
                "Ranks compared".center(widths[1]),
                "L2 norm of difference".center(widths[2]),
            )
        )
        print("-" * table_width)
        for i in range(min(len(l2_diffs), summary_rows)):
            print(
                "|{}|{}|{}|".format(
                    l2_diffs[i][1].center(widths[0]),
                    "{} <-> {}".format(l2_diffs[i][2], l2_diffs[i][3]).center(
                        widths[1]
                    ),
                    f"{l2_diffs[i][0]:.4e}".center(widths[2]),
                )
            )
        print("-" * table_width)

    barrier()
    return consistent
