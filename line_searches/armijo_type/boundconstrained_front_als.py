import numpy as np

from line_searches.armijo_type.als import ALS
from line_searches.line_search_utils.pareto_utils import exists_sufficiently_dominating_point

from problems.problem import Problem


class BoundconstrainedFrontALS(ALS):
    """
    Class for the Bound-constrained Front Armijo-Type Line Search

    The main functions are:
        - Initialize a Bound-constrained Front Armijo-Type Line Search instance;
        - Find a valid step size given a problem solution and a descent direction at it.
    """

    def __init__(self, alpha_0: float, delta: float, beta: float, min_alpha: float):
        """
        Initialize a Bound-constrained Front Armijo-Type Line Search instance
        :param alpha_0: it indicates the initial step size for the Bound-constrained Front Armijo-Type Line Search
        :param delta: it indicates the coefficient for the step size contraction
        :param beta: it indicates the coefficient for the sufficient decrease condition
        :param min_alpha: it indicates the minimum value of alpha that is considered by the Bound-constrained Front Armijo-Type Line Search; after that, the line search fails returning a null step size
        """

        ALS.__init__(self, alpha_0, delta, beta, min_alpha)

    def search(self, problem: Problem, x: np.array, f_list: np.array, d: np.array, theta: float, I: np.array):
        """
        Find a valid step size given a problem solution and a descent direction at it
        :param problem: the considered problem
        :param x: the problem solution
        :param f_list: an array containing the current solutions values in the objectives space
        :param d: the descent direction
        :param theta: the optimal value of the search direction problem given x (for instance, see fmopg.py)
        :param I: the subset of the objective functions indices to be considered
        :return: the step size
        """

        alpha = self._alpha_0

        # Given the current value for alpha, the new point new_x is computed. #

        new_x = x + alpha * d
        new_f = problem.evaluate_functions(new_x)
        f_eval = 1

        """
        In order to be accepted, the current value for alpha must lead to a point new_x such that:
            - it is feasible;
            - the objective functions values calculated at it are neither infinity nor Not A Number;
            - Among the current solutions, there is none that sufficiently dominates it;
            - alpha is greater than min_alpha.
        """

        while (not problem.check_point_feasibility(new_x) or np.isnan(new_f).any() or np.isinf(new_f).any() or exists_sufficiently_dominating_point(new_f[I], f_list[:, I], alpha, self._beta, theta)) and alpha > self._min_alpha:
            alpha *= self._delta
            new_x = x + alpha * d
            new_f = problem.evaluate_functions(new_x)
            f_eval += 1

        # if there is no valid alpha greater than min_alpha, the line search fails returning a null step size. #

        if alpha <= self._min_alpha:
            alpha = 0
            return None, None, alpha, f_eval

        return new_x, new_f, alpha, f_eval
