import time
import numpy as np

from Algorithms.Memetic.Memetic_Algorithm import Memetic_Algorithm
from Algorithms.Genetic.NSGA_II import NSGA_II

from Problems.Problem import Problem


class NSMA(NSGA_II, Memetic_Algorithm):
    """
    Class for the NSMA algorithm

    The main functions are:
        - Initialize a NSMA instance;
        - Execute the algorithm starting from a set of initial points of the problem at hand;
        - Compute the surrogate bounds based on the current population.

    Notes:  Like NSGA-II, NSMA does not require any selection operator.
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
                 pop_size: int,
                 crossover_probability: float,
                 crossover_eta: float,
                 mutation_eta: float,
                 shift: float,
                 crowding_quantile: float,
                 n_opt: int,
                 FMOPG_max_iter: int,
                 theta_for_stationarity: float,
                 theta_tol: float,
                 theta_dec_factor: float,
                 gurobi: bool,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 ALS_alpha_0: float,
                 ALS_delta: float,
                 ALS_beta: float,
                 ALS_min_alpha: float):
        """
        Initialize a NSMA instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the Verbose_System instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the Verbose_System instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each algorithm iteration (see Graphical_Plot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        :param pop_size: size of the population
        :param crossover_probability: see NSGA_II.py
        :param crossover_eta: see NSGA_II.py
        :param mutation_eta: see NSGA_II.py
        :param shift: shift value used to calculate the surrogate bounds
        :param crowding_quantile: requested quantile of the finite crowding distances related to the rank-0 feasible points
        :param n_opt: every n_opt iterations, the FMOPG algorithm is called on some points (see the function search (np.array, np.array, Problem))
        :param FMOPG_max_iter: maximum number of iterations for the FMOPG algorithm
        :param theta_for_stationarity: it indicates the tolerance after which a point is considered Pareto-stationary for the NSMA algorithm; it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param theta_tol: it indicates the tolerance after which a point is considered Pareto-stationary for the FMOPG algorithm; it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param theta_dec_factor: it indicates the coefficient for the theta_tol value contraction
        :param gurobi: if set to True, the Gurobi Optimizer is used to solve the search direction problem
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        :param ALS_alpha_0: it indicates the initial step size for the Armijo-Type Line Search
        :param ALS_delta: it indicates the coefficient for the step size contraction
        :param ALS_beta: it indicates the coefficient for the sufficient decrease condition
        :param ALS_min_alpha: it indicates the minimum value of alpha that is considered by the Armijo-Type Line Search; after that, the line search fails returning a null step size

        Notes:  theta_tol must be smaller than theta_for_stationarity (both are negative numbers).
                In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
                For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
        """

        NSGA_II.__init__(self,
                         max_iter,
                         max_time,
                         max_f_evals,
                         verbose,
                         verbose_interspace,
                         plot_pareto_front,
                         plot_pareto_solutions,
                         plot_dpi,
                         pop_size,
                         crossover_probability,
                         crossover_eta,
                         mutation_eta)

        Memetic_Algorithm.__init__(self,
                                   max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                   pop_size,
                                   theta_tol,
                                   gurobi, gurobi_method, gurobi_verbose, ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                   crowding_quantile, n_opt, theta_for_stationarity, theta_dec_factor,
                                   'FMOPG',
                                   {'theta_tol': theta_tol, 'gurobi': gurobi, 'gurobi_method': gurobi_method, 'gurobi_verbose': gurobi_verbose, 'ALS_alpha_0': ALS_alpha_0, 'ALS_delta': ALS_delta, 'ALS_beta': ALS_beta, 'ALS_min_alpha': ALS_min_alpha, 'FMOPG_max_iter': FMOPG_max_iter, 'max_time': max_time, 'max_f_evals': max_f_evals})

        self.__shift = shift

    def search(self, p_list: np.array, f_list: np.array, problem: Problem):
        """
        Execution of the algorithm, given some initial points of the problem at hand
        :param p_list: initial problem solutions
        :param f_list: related points in the objectives space
        :param problem: the considered problem
        :return: the new solutions (p_list, f_list) and the elapsed time
        """

        self.updateStoppingConditionCurrentValue('max_time', time.time())

        m = f_list.shape[1]

        # Return a list of every possible subset of objective functions indices. #

        objectives_powerset = self.objectivesPowerset(m)

        # Computation of the rank, crowding distance and constraint violation for each problem solution. #
        # Every initial point is considered not epsilon-Pareto-stationary. #

        constraint_violations = np.array([[sum([constraint if constraint > 0 else 0 for constraint in problem.evaluateConstraints(p)])] for p in p_list])
        eps_pareto_stationarity = np.array([[False] * len(objectives_powerset)] * p_list.shape[0])
        p_list, f_list, constraint_violations, rank_list, crowding_list, eps_pareto_stationarity, _ = self._survival_strategy.getSurvivals(p_list, f_list, constraint_violations, eps_pareto_stationarity)

        self.showFigure(p_list, f_list)

        threshold_crowding_distance = None

        while not self.evaluateStoppingConditions():

            # Verbose_System: the metrics specific for NSMA are the maximum finite rank, the maximum rank (including the infinity values) and the minimum crowding distance. #

            self.outputData(f_list, max_finite_rank=max(rank_list[np.isfinite(rank_list)]), max_rank=max(rank_list.flatten()), min_crowding_dist=min(crowding_list.flatten()))

            # Computation of the surrogate bounds to find the offsprings. #

            surrogate_lb, surrogate_ub = self.getSurrogateBounds(p_list, problem)

            # Get the offsprings from the current population. #

            p_list_off = self.getOffsprings(p_list, f_list, constraint_violations, crowding_list, problem, surrogate_lb=surrogate_lb, surrogate_ub=surrogate_ub)

            if len(p_list_off) != 0:
                f_list_off = np.zeros((p_list_off.shape[0], problem.m))
                for index_p_off in range(p_list_off.shape[0]):
                    f_list_off[index_p_off, :] = problem.evaluateFunctions(p_list_off[index_p_off, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', p_list_off.shape[0])

                constraint_violations_off = np.array([[sum([constraint if constraint > 0 else 0 for constraint in problem.evaluateConstraints(p_off)])] for p_off in p_list_off])

                p_list = np.concatenate((p_list, p_list_off), axis=0)
                f_list = np.concatenate((f_list, f_list_off), axis=0)
                constraint_violations = np.concatenate((constraint_violations, constraint_violations_off), axis=0)

                # Every new offspring is considered not epsilon-Pareto-stationarity. #

                eps_pareto_stationarity = np.concatenate((eps_pareto_stationarity, np.array([[False] * len(objectives_powerset)] * p_list_off.shape[0])), axis=0)

                # Given the current population and the offsprings, the survival strategy is used to get at most pop_size survivals. #

                p_list, f_list, constraint_violations, rank_list, crowding_list, eps_pareto_stationarity, threshold_crowding_distance = self._survival_strategy.getSurvivals(p_list, f_list, constraint_violations, eps_pareto_stationarity)

            self.showFigure(p_list, f_list)

            # Every n_opt iterations, the FMOPG algorithm is called on some points. #

            if self.getStoppingConditionCurrentValue('max_iter') % self._n_opt == 0:
                optimization_success = False
                one_tried_optimization = False
                theta_fault = True

                for p in range(p_list.shape[0]):

                    if self.evaluateStoppingConditions():
                        break

                    """
                    The points to optimize through the FMOPG algorithm are the ones with:
                        - rank 0;
                        - no constraint violation;
                        - a crowding-distance that is higher than or equal to the threshold_crowding_distance (the requested quantile of the finite crowding distances related to the rank-0 feasible points).
                    """

                    if rank_list[p] == 0 and constraint_violations[p] == 0 and crowding_list[p] >= threshold_crowding_distance:

                        # The selected points are optimized w.r.t. each subset of the objective functions indices, if they are not epsilon-Pareto-stationary for that subset. #

                        for index_I_p, I_p in enumerate(objectives_powerset):

                            if not eps_pareto_stationarity[p, index_I_p]:

                                if self.evaluateStoppingConditions():
                                    break

                                # In order to be optimized for a subset of the objective functions indices, the point must be not dominated by other points w.r.t. that subset. #

                                if self.existsDominatingPoint(f_list[p, I_p, ], np.delete(f_list, p, 0)[:, I_p, ]):
                                    continue

                                one_tried_optimization = True

                                n_current_points = p_list.shape[0]

                                # theta_tol indicates the tolerance after which a point is considered Pareto-stationary for the FMOPG algorithm. #

                                self._local_search_optimizer.resetStoppingConditionsCurrentValues(self._theta_tol)
                                p_list, f_list, theta_array = self._local_search_optimizer.search(p_list, f_list, problem, index_initial_point=p, I=I_p)

                                self.updateStoppingConditionCurrentValue('max_f_evals', self._local_search_optimizer.getStoppingConditionCurrentValue('max_f_evals'))

                                # theta_for_stationarity indicates the tolerance after which a point is considered Pareto-stationary for the NSMA algorithm. #

                                if theta_array[0] >= self._theta_for_stationarity:
                                    eps_pareto_stationarity[p, index_I_p] = True

                                if p_list.shape[0] > n_current_points:

                                    # At least one new point is obtained by the FMOPG algorithm. #

                                    optimization_success = True
                                    theta_fault = False

                                    for index_new_point in range(n_current_points, p_list.shape[0]):
                                        constraint_violations = np.concatenate((constraint_violations, np.array([[sum([constraint if constraint > 0 else 0 for constraint in problem.evaluateConstraints(p_list[index_new_point, :])])]])), axis=0)
                                        eps_pareto_stationary_tmp = np.array([[False] * len(objectives_powerset)])
                                        if theta_array[index_new_point - n_current_points + 1] >= self._theta_for_stationarity:
                                            eps_pareto_stationary_tmp[0, index_I_p] = True
                                        eps_pareto_stationarity = np.concatenate((eps_pareto_stationarity, eps_pareto_stationary_tmp), axis=0)

                                else:
                                    if (theta_array < self._theta_tol).any():
                                        theta_fault = False

                if optimization_success:

                    # If at least a point from the FMOPG algorithm is obtained, NSMA uses the survival strategy to get at most pop_size survivals. #

                    p_list, f_list, constraint_violations, rank_list, crowding_list, eps_pareto_stationarity, _ = self._survival_strategy.getSurvivals(p_list, f_list, constraint_violations, eps_pareto_stationarity)

                # If at least a point from the FMOPG algorithm is obtained or no point is generated because theta_tol is too small (negative number), then theta_tol is contracted through the coefficient theta_dec_factor. #

                self._theta_tol *= self._theta_dec_factor if optimization_success or (one_tried_optimization and theta_fault) else 1

                self.showFigure(p_list, f_list)

            self.addToStoppingConditionCurrentValue('max_iter', 1)

        self.closeFigure()

        self.outputData(f_list, max_finite_rank=max(rank_list[np.isfinite(rank_list)]), max_rank=max(rank_list.flatten()), max_crowding_dist=min(crowding_list.flatten()))

        return p_list, f_list, time.time() - self.getStoppingConditionCurrentValue('max_time')

    def getSurrogateBounds(self, p_list: np.array, problem: Problem):
        """
        Compute the surrogate bounds based on the current population
        :param p_list: population solutions
        :param problem: the problem at hand
        :return: the surrogate bounds
        """

        surrogate_lb = np.max(np.array([problem.lb, np.min(p_list, axis=0) - self.__shift]), axis=0)
        surrogate_ub = np.min(np.array([problem.ub, np.max(p_list, axis=0) + self.__shift]), axis=0)

        return surrogate_lb, surrogate_ub
