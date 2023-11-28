import torch
from inspect import getframeinfo, stack


class MemStat:
    """
    A class to print gpu memory usage statistics. Use inline with a statement describing the current
    execution step.
    """

    def __init__(self) -> None:
        self.start_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB

    def reset(self):
        """
        Resets the starting memory to the current GPU memory usage.
        """
        self.start_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB

    def __call__(self, name=None):
        """
        Prints the memory usage statistics.

        Args:
            name (str, optional): Name of the current execution step. Defaults to None.

        Example:
            > m = MemStat()
            > m("Part A")
            main.py L2: MemStat Part A: 1.63 GB (+0.41 GB)
            > m()
            main.py L3: MemStat: 45.61 GB (+43.59 GB)
        """
        if name is None:
            name = ""

        caller = getframeinfo(stack()[1][0])
        mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB

        diff = mem - self.start_mem
        diff_string = (
            "+{:.2f} GB".format(diff) if diff > 0 else "{:.2f} GB".format(diff)
        )

        print(
            "{} L{}: MemStat {}: {:.2f} GB ({})".format(
                caller.filename, caller.lineno, name, mem, diff_string
            )
        )
        self.start_mem = mem


import torch
from inspect import getframeinfo, stack


class MemStatBlock:
    """
    A class to print GPU memory usage statistics. Use inline with a statement describing the current
    execution step.

    Usage:
    ------
    with MemStatBlock("Forward Pass"):
        # Perform forward pass operations

    with MemStatBlock("Backward Pass"):
        # Perform backward pass operations

    """

    def __init__(self, name=None) -> None:
        self.name = name

    def __enter__(self):
        self.start_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB
        print(self.start_mem)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        end_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB
        print(end_mem)
        if self.name is None:
            self.name = ""
        caller = getframeinfo(stack()[1][0])

        diff = end_mem - self.start_mem
        diff_string = (
            "+{:.2f} GB".format(diff) if diff > 0 else "{:.2f} GB".format(diff)
        )
        print(
            "{} L{}: MemStats {}: {:.2f} GB ({})".format(
                caller.filename, caller.lineno, self.name, end_mem, diff_string
            )
        )
