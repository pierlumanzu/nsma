from abc import abstractmethod
import numpy as np

from problems.problem import Problem


class ALS:
    """
    Abstract class for the Armijo-Type Line Search

    The main functions are:
        - Initialize an Armijo-Type Line Search instance;
        - Find a valid step size given a problem solution and a descent direction at it (to be implemented).
    """

    def __init__(self, alpha_0: float, delta: float, beta: float, min_alpha: float):
        """
        Initialize an Armijo-Type Line Search instance
        :param alpha_0: it indicates the initial step size for the Armijo-Type Line Search
        :param delta: it indicates the coefficient for the step size contraction
        :param beta: it indicates the coefficient for the sufficient decrease condition
        :param min_alpha: it indicates the minimum value of alpha that is considered by the Armijo-Type Line Search; after that, the line search fails returning a null step size
        """

        self._alpha_0 = alpha_0
        self._delta = delta
        self._beta = beta

        self._min_alpha = min_alpha

    @abstractmethod
    def search(self, problem: Problem, x: np.array, f_list: np.array, d: np.array, theta: float, I: np.array):
        """
        Find a valid step size given a problem solution and a descent direction at it (to be implemented)
        :param problem: the considered problem
        :param x: the problem solution
        :param f_list: an array containing the current solutions values in the objectives space
        :param d: the descent direction
        :param theta: the optimal value of the search direction problem given x (for instance, see fmopg.py)
        :param I: the subset of the objective functions indices to be considered
        :return: the step size
        """

        return
