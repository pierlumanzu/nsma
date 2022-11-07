import numpy as np
import tensorflow as tf

from Problems.MAN.MAN_Class import MAN

# The user is referred to the article for more details about the MAN problem. #


class MAN_1(MAN):
    """
    Class for the MAN_1 problem

    The main functions are:
        - Initialize a MAN_1 problem instance;
        - Return the name of the problem.
    """

    def __init__(self, n: int):
        """
        Initialize a MAN_1 problem instance
        :param n: the problem dimensionality

        Notes:  It raises an AssertionError if the given problem dimensionality is less than 2.
        """

        assert n >= 2
        MAN.__init__(self, n)

        # Objective functions definition. #

        self.objectives = [
            tf.reduce_sum([(self._z[i] - (i + 1)) ** 2 for i in range(self.n)]) / self.n ** 2,
            tf.reduce_sum([tf.exp(-self._z[i]) + self._z[i] for i in range(self.n)])
        ]

        # Bound constraints definition. #

        self.lb = -10 ** 4 * np.ones(self.n)
        self.ub = 10 ** 4 * np.ones(self.n)

    @staticmethod
    def name():
        """
        Return the name of the problem
        :return: the name of the problem
        """

        return 'MAN_1'
