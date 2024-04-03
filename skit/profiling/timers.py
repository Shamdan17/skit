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
    Tick Part A took: 1.63 ms
    > t()
    Tick took: 45.61 ms
    ...

    Methods:
    - __init__(self, verbose=True): Initializes the Timer object.
    - reset(self): Resets the timer by updating the start time to the current time.
    - tick(self, name=None): Prints the time elapsed since the last tick.
    - __call__(self, name=None): Prints the time elapsed since the last tick.

    Attributes:
    - verbose (bool): Whether to print verbose output or not. Default is True.
    """

    def __init__(self, verbose=True):
        """
        Initialize the Timer object.

        Parameters:
        - verbose (bool): Whether to print verbose output or not. Default is True. If false, only the
        time elapsed since the last tick will be returned without printing.
        """
        self.reset()
        self.verbose = verbose

    def reset(self):
        """
        Resets the timer by updating the start time to the current time.

        Call this function if you want to reset the timer without printing
        """
        self.start = timeit.default_timer()

    def tick(self, name=None):
        """
        Prints the time elapsed since the last tick. You can optionally
        provide a name for the tick. The same function can also be called
        directly.

        Usage:
        > t = Ticker()
        > t.tick("Part A")
        Tick Part A took: 1.63 ms
        > t()
        Tick took: 45.61 ms

        Parameters:
        - name (str): The name for the tick. Default is None.

        Returns:
        - tick_time (float): The time elapsed since the last tick in milliseconds.
        """
        if name is None:
            name = ""
        tick_time = (timeit.default_timer() - self.start) * 1000.0
        if self.verbose:
            print("Tick {} took: {:.2f} ms.".format(name, tick_time))
        self.start = timeit.default_timer()
        return tick_time

    def __call__(self, name=None):
        """
        Prints the time elapsed since the last tick.

        Usage:
        > t = Ticker()
        > t("Part A")
        Tick Part A took: 1.63 ms
        > t()
        Tick took: 45.61 ms

        Parameters:
        - name (str): The name for the tick. Default is None.

        Returns:
        - tick_time (float): The time elapsed since the last tick in milliseconds.
        """
        return self.tick(name)
