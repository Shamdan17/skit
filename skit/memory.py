import torch
from inspect import getframeinfo, stack


class MemStats:
    """
    A class to print gpu memory usage statistics. Use inline with a statement describing the current
    execution step.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, name=None):
        """
        Prints the memory usage statistics.
        Usage:
        > m = MemStats()
        > m("Part A")
        main.py L2: MemStats Part A: 1.63 GB
        > m()
        main.py L3: MemStats: 45.61 GB
        """
        if name is None:
            name = ""

        caller = getframeinfo(stack()[1][0])
        mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # in GB

        print(
            "{} L{}: MemStats {}: {:.2f} GB".format(
                caller.filename, caller.lineno, name, mem
            )
        )
