from abc import abstractmethod
import numpy as np

from problems.problem import Problem


class DDS:
    """
    Abstract class for the search direction problem

    The main functions are:
        - Initialize a class instance to solve the search direction problem;
        - Find the descent direction at a specific solution for the problem at hand (to be implemented).
    """

    def __init__(self):
        """
        Initialize a class instance to solve the search direction problem
        """

        pass

    @abstractmethod
    def compute_direction(self, problem: Problem, Jac: np.array, x_p: np.array = None):
        """
        Find the descent direction at a specific solution for the problem at hand (to be implemented)
        :param problem: the considered problem
        :param Jac: the Jacobian matrix related to the objective functions calculated at x_p
        :param x_p: the problem solution
        :return: the descent direction
        """

        return
