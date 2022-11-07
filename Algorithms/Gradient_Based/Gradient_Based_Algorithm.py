import numpy as np
from abc import ABC
from itertools import chain, combinations

from Algorithms.Algorithm import Algorithm

from Direction_Solvers.Direction_Solver_Factory import Direction_Descent_Factory
from Line_Searches.Line_Search_Factory import Line_Search_Factory


class Gradient_Based_Algorithm(Algorithm, ABC):
    """
    Abstract class for gradient-based algorithms

    The main functions are:
        - Initialize a gradient-based algorithm instance;
        - Return the powerset of the objective functions indices;
        - Check for a specific solution if there is another one in a given array that dominates it;
        - Check for each current solution if there is one in the new array that dominates it.
    """

    def __init__(self,
                 max_iter: int,
                 max_time: float,
                 max_f_evals: int,
                 verbose: bool,
                 verbose_interspace: int,
                 plot_pareto_front: bool,
                 plot_pareto_solutions: bool,
                 plot_dpi: int,
                 theta_tol: float,
                 gurobi: bool,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 ALS_alpha_0: float,
                 ALS_delta: float,
                 ALS_beta: float,
                 ALS_min_alpha: float,
                 name_DDS: str = None,
                 name_ALS: str = None):
        """
        Initialize a gradient-based algorithm instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the Verbose_System instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the Verbose_System instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each algorithm iteration (see Graphical_Plot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        :param theta_tol: it indicates the tolerance after which a point is considered Pareto-stationary; it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param gurobi: if set to True, the Gurobi Optimizer is used to solve the search direction problem
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        :param ALS_alpha_0: it indicates the initial step size for the Armijo-Type Line Search
        :param ALS_delta: it indicates the coefficient for the step size contraction
        :param ALS_beta: it indicates the coefficient for the sufficient decrease condition
        :param ALS_min_alpha: it indicates the minimum value of alpha that is considered by the Armijo-Type Line Search; after that, the line search fails returning a null step size
        :param name_DDS: the name of the class for the search direction problem
        :param name_ALS: the name of the Armijo-Type Line Search

        Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
                For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
        """

        Algorithm.__init__(self,
                           max_iter,
                           max_time,
                           max_f_evals,
                           verbose,
                           verbose_interspace,
                           plot_pareto_front,
                           plot_pareto_solutions,
                           plot_dpi)

        self._theta_tol = theta_tol

        self._direction_solver = Direction_Descent_Factory.getDirectionCalculator(name_DDS, gurobi, gurobi_method, gurobi_verbose) if name_DDS is not None else None
        self._line_search = Line_Search_Factory.getLineSearch(name_ALS, ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha) if name_ALS is not None else None

    @staticmethod
    def objectivesPowerset(m: int):
        """
        Return the powerset of the objective functions indices
        :param m: number of objective functions
        :return: the powerset of the objective functions indices
        """

        s = list(range(m))
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1, 0, -1)))

    @staticmethod
    def existsDominatingPoint(new_f: np.array, f_list: np.array):
        """
        Check for a specific solution if there is another one in a given array that dominates it
        :param new_f: the solution values in the objectives space
        :param f_list: the array
        :return: a boolean value indicating if the new solution is dominated or not
        """

        if np.isnan(new_f).any():
            return True

        n_obj = len(new_f)

        new_f = np.reshape(new_f, (1, n_obj))
        dominance_matrix = f_list - new_f

        is_dominated = (np.sum(dominance_matrix <= 0, axis=1) == n_obj).any()

        return is_dominated

    @staticmethod
    def fastNonDominatedFilter(curr_f_list: np.array, new_f_list: np.array):
        """
        Check for each current solution if there is one in the new array that dominates it
        :param curr_f_list: array of current solutions values in the objectives space
        :param new_f_list: new array
        :return: an array indicating for each current solution if there is one in the new array that dominates it (False) or there is not (True)
        """

        n_new_points, m = new_f_list.shape
        efficient = np.array([True] * curr_f_list.shape[0])

        for i in range(n_new_points):
            dominance_matrix = curr_f_list - np.reshape(new_f_list[i, :], newshape=(1, m))
            dominated_idx = np.sum(dominance_matrix >= 0, axis=1) == m

            assert len(dominated_idx.shape) == 1
            dom_indices = np.where(dominated_idx)[0]

            if len(dom_indices) > 0:
                efficient[dom_indices] = False

        return efficient
