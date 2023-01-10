import time
import numpy as np
from abc import abstractmethod
from typing import Union

from algorithms.algorithm_utils.graphical_plot import GraphicalPlot
from algorithms.algorithm_utils.stopping_condition import StoppingCondition
from algorithms.algorithm_utils.verbose_system import VerboseSystem

from problems.problem import Problem


class Algorithm:
    """
        Abstract class for generic algorithms

        The main functions are:
            - Initialize an Algorithm instance;
            - Execute the Algorithm starting from a set of initial points of the problem at hand (to be implemented);
            - Add a new stopping condition (see StoppingCondition.py);
            - Add a value to the current one(s) of a stopping condition;
            - Update the current value of a stopping condition;
            - Update the reference value of a stopping condition;
            - Get the current value of a stopping condition;
            - Check if one of the stopping conditions is satisfied;
            - Call the VerboseSystem instance to show the standard metrics and the ones specific for the Algorithm at hand (argument verbose must be activated);
            - Trigger the GraphicalPlot instance such that the Pareto front plot and the Pareto solutions one are shown (argument plot_pareto_front must be activated; in order also to plot the Pareto solutions whenever is possible, plot_pareto_solutions must be activated; for more details, see parser_management.py);
            - Through the GraphicalPlot instance, close the figure of the plots.
        """

    def __init__(self,
                 max_iter: int,
                 max_time: float,
                 max_f_evals: int,
                 verbose: bool,
                 verbose_interspace: int,
                 plot_pareto_front: bool,
                 plot_pareto_solutions: bool,
                 plot_dpi: int):
        """
        Initialize the Algorithm instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the VerboseSystem instance is used during the Algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the VerboseSystem instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each Algorithm iteration (see GraphicalPlot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each Algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        """

        self.__stopping_conditions = {'max_iter': [StoppingCondition(max_iter, equal_required=True), 0],
                                      'max_time': [StoppingCondition(max_time * 60 if max_time is not None else None, equal_required=True), 0],
                                      'max_f_evals': [StoppingCondition(max_f_evals, equal_required=True), 0]}

        self._verbose = verbose
        self.__verbose_system = VerboseSystem(verbose_interspace) if self._verbose else None

        self._plot_pareto_front = plot_pareto_front
        self.__graphical_plot = GraphicalPlot(plot_pareto_solutions, plot_dpi) if self._plot_pareto_front else None

    @abstractmethod
    def search(self, p_list: np.array, f_list: np.array, problem: Problem):
        """
        Execute the Algorithm starting from a set of initial points of the problem at hand (to be implemented)
        :param p_list: initial problem solutions
        :param f_list: related points in the objectives space
        :param problem: the considered problem
        :return: the new solutions (p_list, f_list) and the elapsed time
        """

        pass

    def add_stopping_condition(self, condition_name: str, reference_value: float, current_value: Union[float, int, np.float64, np.ndarray], smaller_value_required: bool = False, equal_required: bool = False):
        """
        Add a new stopping condition (see StoppingCondition.py)
        :param condition_name: the name of the stopping condition
        :param reference_value: the reference value of the stopping condition
        :param current_value: the current value of the stopping condition
        :param smaller_value_required: if set to True, the current value(s) must be smaller than the reference value in order to stop the execution of the Algorithm
        :param equal_required: if set to True, the current value(s) can be also equal to the reference value in order to stop the execution of the Algorithm

        Notes:  It raises an AssertionError if a stopping condition with the same name as the one proposed by the user already exists.
        """

        assert condition_name not in self.__stopping_conditions.keys()
        self.__stopping_conditions[condition_name] = [StoppingCondition(reference_value, smaller_value_required=smaller_value_required, equal_required=equal_required), current_value]

    def add_to_stopping_condition_current_value(self, condition_name: str, adding_value: float):
        """
        Add a value to the current one(s) of a stopping condition
        :param condition_name: the name of the stopping condition
        :param adding_value: value to add to the stopping condition current value(s)
        """

        self.__stopping_conditions[condition_name][1] += adding_value

    def update_stopping_condition_current_value(self, condition_name: str, current_value: Union[float, int, np.float64, np.ndarray]):
        """
        Update the current value of a stopping condition
        :param condition_name: name of the stopping condition
        :param current_value: new stopping condition current value
        """

        self.__stopping_conditions[condition_name][1] = np.copy(current_value) if type(current_value) == np.ndarray else current_value

    def update_stopping_condition_reference_value(self, condition_name: str, reference_value: float):
        """
        Update the reference value of a stopping condition
        :param condition_name: name of the stopping condition
        :param reference_value: new stopping condition reference value
        """

        self.__stopping_conditions[condition_name][0].update_reference_value(reference_value)

    def get_stopping_condition_current_value(self, condition_name: str):
        """
        Get the current value of a stopping condition
        :param condition_name: name of the stopping condition
        :return: the stopping condition current value
        """

        return self.__stopping_conditions[condition_name][1]

    def evaluate_stopping_conditions(self):
        """
        Check if one of the stopping conditions is satisfied
        :return: a boolean value indicating if a stopping condition is satisfied
        """

        for condition_name in self.__stopping_conditions.keys():
            stopping_condition, current_value = self.__stopping_conditions[condition_name]
            if condition_name == 'max_time':
                current_value = time.time() - current_value
            if stopping_condition.evaluate_stopping_condition(current_value):
                return True

        return False

    def output_data(self, f_list: np.array, **kwargs):
        """
        Call the VerboseSystem instance to show the standard metrics and the ones specific for the Algorithm at hand
        :param f_list: points in the objectives space
        :param kwargs: metrics specific for the Algorithm at hand with related values

        Notes:  Argument verbose must be activated
        """

        if self._verbose:
            self.__verbose_system.output_data(self.__stopping_conditions['max_iter'][1],
                                              time.time() - self.__stopping_conditions['max_time'][1],
                                              self.__stopping_conditions['max_f_evals'][1],
                                              f_list,
                                              **kwargs)

    def show_figure(self, p_list: np.array, f_list: np.array, hold_still: bool = False):
        """
        Trigger the GraphicalPlot instance such that the Pareto front plot and the Pareto solutions one are shown
        :param p_list: Pareto solutions
        :param f_list: Pareto front points
        :param hold_still: if set to True, the execution is stopped; once that the plot window is closed, the execution continues

        Notes:  Argument plot_pareto_front must be activated.
                In order also to plot the Pareto solutions whenever is possible, plot_pareto_solutions must be activated.
                For more details, see parser_management.py.
        """

        if self._plot_pareto_front:
            self.__graphical_plot.show_figure(p_list, f_list, hold_still)

    def close_figure(self):
        """
        Through the GraphicalPlot instance, close the figure of the plots
        """

        if self._plot_pareto_front:
            self.__graphical_plot.close_figure()

