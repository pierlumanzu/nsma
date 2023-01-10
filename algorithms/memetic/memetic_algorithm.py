from abc import ABC
from typing import Union
import numpy as np

from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm
from algorithms.memetic.memetic_utils.survival_strategy import MemeticSurvivalStrategy
from algorithms.gradient_based.local_search_algorithms.local_search_algorithm_factory import LocalSearchAlgorithmFactory


class MemeticAlgorithm(GeneticAlgorithm, GradientBasedAlgorithm, ABC):
    """
    Abstract class for memetic algorithms

    The main functions are:
        - Initialize a memetic algorithm instance;
        - Add a specified value to the current one(s) of a stopping condition (see StoppingCondition.py);
        - Update the current value of a stopping condition.
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
                 theta_tol: float,
                 gurobi: bool,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 ALS_alpha_0: float,
                 ALS_delta: float,
                 ALS_beta: float,
                 ALS_min_alpha: float,
                 crowding_quantile: float,
                 n_opt: int,
                 theta_for_stationarity: float,
                 theta_dec_factor: float,
                 name_local_search_algorithm: str,
                 args_local_search_algorithm: dict):
        """
        Initialize a memetic algorithm instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the VerboseSystem instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the VerboseSystem instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each Algorithm iteration (see GraphicalPlot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each Algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        :param pop_size: size of the population
        :param theta_tol: it indicates the tolerance after which a point is considered Pareto-stationary for the local search gradient-based Algorithm (for instance, see nsma.py); it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param gurobi: if set to True, the Gurobi Optimizer is used to solve the search direction problem
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        :param ALS_alpha_0: it indicates the initial step size for the Armijo-Type Line Search
        :param ALS_delta: it indicates the coefficient for the step size contraction
        :param ALS_beta: it indicates the coefficient for the sufficient decrease condition
        :param ALS_min_alpha: it indicates the minimum value of alpha that is considered by the Armijo-Type Line Search; after that, the line search fails returning a null step size
        :param crowding_quantile: requested quantile of the finite crowding distances related to the rank-0 feasible points
        :param n_opt: every n_opt iterations, a local search gradient-based Algorithm is called on some points (for instance, see nsma.py)
        :param theta_for_stationarity: it indicates the tolerance after which a point is considered Pareto-stationary for the memetic Algorithm; it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param theta_dec_factor: it indicates the coefficient for the theta_tol value contraction
        :param name_local_search_algorithm: name of the local search gradient-based Algorithm used inside the memetic Algorithm
        :param args_local_search_algorithm: arguments to take as inputs in the local search Algorithm constructor

        Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
                For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
                theta_tol must be smaller than theta_for_stationarity (both are negative numbers).
        """

        GeneticAlgorithm.__init__(self,
                                  max_iter,
                                  max_time,
                                  max_f_evals,
                                  verbose,
                                  verbose_interspace,
                                  plot_pareto_front,
                                  plot_pareto_solutions,
                                  plot_dpi,
                                  pop_size)

        self._survival_strategy = MemeticSurvivalStrategy(pop_size, crowding_quantile)

        GradientBasedAlgorithm.__init__(self,
                                        max_iter,
                                        max_time,
                                        max_f_evals,
                                        verbose,
                                        verbose_interspace,
                                        plot_pareto_front,
                                        plot_pareto_solutions,
                                        plot_dpi,
                                        theta_tol,
                                        gurobi,
                                        gurobi_method,
                                        gurobi_verbose,
                                        ALS_alpha_0,
                                        ALS_delta,
                                        ALS_beta,
                                        ALS_min_alpha)

        self._n_opt = n_opt
        self._theta_for_stationarity = theta_for_stationarity
        self._theta_dec_factor = theta_dec_factor

        self._local_search_optimizer = LocalSearchAlgorithmFactory.get_algorithm(name_local_search_algorithm, args_local_search_algorithm)

    def add_to_stopping_condition_current_value(self, condition_name: str, adding_value: float):
        """
        Add a specified value to the current one(s) of a stopping condition (see StoppingCondition.py)
        :param condition_name: name of the stopping condition
        :param adding_value: value to add to the stopping condition current value(s)

        Notes:  Regarding the stopping conditions 'max_time' and 'max_f_evals', it also adds the specified value to the current one of the homonym stopping conditions related to the local search gradient-based Algorithm.
        """

        GeneticAlgorithm.add_to_stopping_condition_current_value(self, condition_name, adding_value)
        if condition_name in ['max_time', 'max_f_evals']:
            self._local_search_optimizer.add_to_stopping_condition_current_value(condition_name, adding_value)

    def update_stopping_condition_current_value(self, condition_name: str, current_value: Union[float, int, np.float64, np.ndarray]):
        """
        Update the current value of a stopping condition
        :param condition_name: name of the stopping condition
        :param current_value: new stopping condition current value

        Notes:  Regarding the stopping conditions 'max_time' and 'max_f_evals', it also updates the current value of the homonym stopping conditions related to the local search gradient-based Algorithm.
        """

        GeneticAlgorithm.update_stopping_condition_current_value(self, condition_name, current_value)
        if condition_name in ['max_time', 'max_f_evals']:
            self._local_search_optimizer.update_stopping_condition_current_value(condition_name, current_value)
