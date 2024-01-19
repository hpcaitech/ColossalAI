"""
A class that can be used to calculate the mean of a variable
"""


class AccumulativeMeanVariable:
    """
    A class that calculates the accumulative mean of a variable.
    """

    def __init__(self):
        self._sum = 0
        self._count = 0

    def add(self, value, count_update=1):
        """
        Adds a value to the sum and updates the count.

        Args:
            value (float): The value to be added.
            count_update (int, optional): The amount to update the count by. Defaults to 1.
        """
        self._sum += value
        self._count += count_update

    def get(self):
        """
        Calculates and returns the accumulative mean.

        Returns:
            float: The accumulative mean.
        """
        return self._sum / self._count if self._count > 0 else 0

    def reset(self):
        """
        Resets the sum and count to zero.
        """
        self._sum = 0
        self._count = 0


class AccumulativeMeanMeter:
    """
    A class for calculating and storing the accumulative mean of variables.

    Attributes:
        variable_dict (dict): A dictionary to store the accumulative mean variables.

    Methods:
        add(name, value, count_update=1): Adds a value to the specified variable.
        get(name): Retrieves the accumulative mean value of the specified variable.
        reset(): Resets all the accumulative mean variables to their initial state.
    """

    def __init__(self):
        self.variable_dict = {}

    def add(self, name, value, count_update=1):
        if name not in self.variable_dict:
            self.variable_dict[name] = AccumulativeMeanVariable()
        self.variable_dict[name].add(value, count_update=count_update)

    def get(self, name):
        return self.variable_dict[name].get()

    def reset(self):
        for name in self.variable_dict:
            self.variable_dict[name].reset()
