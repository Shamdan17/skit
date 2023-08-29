import timeit


class BlockTimer:
    """
    A simple timer class that prints the time elapsed for a code block.
    Usage:
    > with BlockTimer("Part A"):
    >     # Code block
    Code block 'Part A' took: 1.63 ms
    > with BlockTimer():
    >     # Code block
    Code block took: 45.61 ms
    """

    def __init__(self, name=None):
        if name:
            self.name = "Code block {}".format(name)
        else:
            self.name = "Code block"

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print("{} took: {:.2f} ms.".format(self.name, self.took))


class Ticker:
    """
    A simple timer class that prints the time elapsed since the last tick.
    Usage:
    > t = Ticker()
    > t("Part A")
    Part A took: 1.63 ms
    > t("Part B")
    Part B took: 45.61 ms
    ...
    > t.reset()
    > t()
    Tick took: 3.09 ms
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the timer.
        Usage:
        > t = Ticker()
        > t.reset()
        """
        self.start = timeit.default_timer()

    def tick(self, name=None):
        """
        Prints the time elapsed since the last tick. You can optionally
        provide a name for the tick. The same function can also be called
        directly
        Usage:
        > t = Ticker()
        > t.tick("Part A")
        Tick Part A took: 1.63 ms
        > t()
        Tick took: 45.61 ms
        """
        if name is None:
            name = "Tick"
        tick_time = (timeit.default_timer() - self.start) * 1000.0
        print("{} took: {:.2f} ms.".format(name, tick_time))
        self.start = timeit.default_timer()

    def __call__(self, name=None):
        """
        Prints the time elapsed since the last tick.
        Usage:
        > t = Ticker()
        > t("Part A")
        Tick Part A took: 1.63 ms
        > t()
        Tick took: 45.61 ms
        """
        self.tick(name)
