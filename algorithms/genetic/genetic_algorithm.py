from abc import abstractmethod
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.genetic.genetic_utils.survival_strategy import GeneticSurvivalStrategy

from problems.problem import Problem


class GeneticAlgorithm(Algorithm):
    """
    Abstract class for genetic algorithms

    The main functions are:
        - Initialize a genetic algorithm instance;
        - Get the solution parents (to be implemented);
        - Execute the crossover operation (to be implemented);
        - Execute the mutation operation (to be implemented);
        - Execute the selection operation (to be implemented);
        - Get the solution offsprings (to be implemented).
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
                 pop_size: int):
        """
        Initialize a genetic algorithm instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the VerboseSystem instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the VerboseSystem instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each algorithm iteration (see GraphicalPlot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        :param pop_size: size of the population
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

        self._pop_size = pop_size
        self._survival_strategy = GeneticSurvivalStrategy(pop_size)

    @abstractmethod
    def get_parents(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, crowding_list: np.array):
        """
        Get the solution parents (to be implemented)
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param crowding_list: an array containing all the crowding distances; each of them is related to a point in p_list
        :return: the solution parents
        """

        pass

    @abstractmethod
    def crossover(self):
        """
        Execute the crossover operation (to be implemented);
        :return: the new solution offsprings
        """

        pass

    @abstractmethod
    def mutation(self):
        """
        Execute the mutation operation (to be implemented);
        :return: the mutated solution offsprings
        """

        pass

    @abstractmethod
    def selection(self):
        """
        Execute the selection operation (to be implemented);
        :return: the selected offsprings to insert in the population
        """

        pass

    @abstractmethod
    def get_offsprings(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, crowding_list: np.array, problem: Problem):
        """
        Get the solution offsprings (to be implemented);
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param crowding_list: an array containing all the crowding distances; each of them is related to a point in p_list
        :param problem: the considered problem
        :return: the solution offsprings
        """

        pass
