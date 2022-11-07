import time
import numpy as np
from abc import abstractmethod
from typing import Union

from Algorithms.Algorithm_Utils.Graphical_Plot import Graphical_Plot
from Algorithms.Algorithm_Utils.Stopping_Condition import Stopping_Condition
from Algorithms.Algorithm_Utils.Verbose_System import Verbose_System

from Problems.Problem import Problem


class Algorithm:
    """
        Abstract class for generic algorithms

        The main functions are:
            - Initialize an algorithm instance;
            - Execute the algorithm starting from a set of initial points of the problem at hand (to be implemented);
            - Add a new stopping condition (see Stopping_Condition.py);
            - Add a value to the current one(s) of a stopping condition;
            - Update the current value of a stopping condition;
            - Update the reference value of a stopping condition;
            - Get the current value of a stopping condition;
            - Check if one of the stopping conditions is satisfied;
            - Call the Verbose_System instance to show the standard metrics and the ones specific for the algorithm at hand (argument verbose must be activated);
            - Trigger the Graphical_Plot instance such that the Pareto front plot and the Pareto solutions one are shown (argument plot_pareto_front must be activated; in order also to plot the Pareto solutions whenever is possible, plot_pareto_solutions must be activated; for more details, see parser_management.py);
            - Through the Graphical_Plot instance, close the figure of the plots.
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
        Initialize the algorithm instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the Verbose_System instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the Verbose_System instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each algorithm iteration (see Graphical_Plot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        """

        self.__stopping_conditions = {'max_iter': [Stopping_Condition(max_iter, equal_required=True), 0],
                                      'max_time': [Stopping_Condition(max_time * 60 if max_time is not None else None, equal_required=True), 0],
                                      'max_f_evals': [Stopping_Condition(max_f_evals, equal_required=True), 0]}

        self._verbose = verbose
        self.__verbose_system = Verbose_System(verbose_interspace) if self._verbose else None

        self._plot_pareto_front = plot_pareto_front
        self.__graphical_plot = Graphical_Plot(plot_pareto_solutions, plot_dpi) if self._plot_pareto_front else None

    @abstractmethod
    def search(self, p_list: np.array, f_list: np.array, problem: Problem):
        """
        Execute the algorithm starting from a set of initial points of the problem at hand (to be implemented)
        :param p_list: initial problem solutions
        :param f_list: related points in the objectives space
        :param problem: the considered problem
        :return: the new solutions (p_list, f_list) and the elapsed time
        """

        pass

    def addStoppingCondition(self, condition_name: str, reference_value: float, current_value: Union[float, int, np.float64, np.ndarray], smaller_value_required: bool = False, equal_required: bool = False):
        """
        Add a new stopping condition (see Stopping_Condition.py)
        :param condition_name: the name of the stopping condition
        :param reference_value: the reference value of the stopping condition
        :param current_value: the current value of the stopping condition
        :param smaller_value_required: if set to True, the current value(s) must be smaller than the reference value in order to stop the execution of the algorithm
        :param equal_required: if set to True, the current value(s) can be also equal to the reference value in order to stop the execution of the algorithm

        Notes:  It raises an AssertionError if a stopping condition with the same name as the one proposed by the user already exists.
        """

        assert condition_name not in self.__stopping_conditions.keys()
        self.__stopping_conditions[condition_name] = [Stopping_Condition(reference_value, smaller_value_required=smaller_value_required, equal_required=equal_required), current_value]

    def addToStoppingConditionCurrentValue(self, condition_name: str, adding_value: float):
        """
        Add a value to the current one(s) of a stopping condition
        :param condition_name: the name of the stopping condition
        :param adding_value: value to add to the stopping condition current value(s)
        """

        self.__stopping_conditions[condition_name][1] += adding_value

    def updateStoppingConditionCurrentValue(self, condition_name: str, current_value: Union[float, int, np.float64, np.ndarray]):
        """
        Update the current value of a stopping condition
        :param condition_name: name of the stopping condition
        :param current_value: new stopping condition current value
        """

        self.__stopping_conditions[condition_name][1] = np.copy(current_value) if type(current_value) == np.ndarray else current_value

    def updateStoppingConditionReferenceValue(self, condition_name: str, reference_value: float):
        """
        Update the reference value of a stopping condition
        :param condition_name: name of the stopping condition
        :param reference_value: new stopping condition reference value
        """

        self.__stopping_conditions[condition_name][0].updateReferenceValue(reference_value)

    def getStoppingConditionCurrentValue(self, condition_name: str):
        """
        Get the current value of a stopping condition
        :param condition_name: name of the stopping condition
        :return: the stopping condition current value
        """

        return self.__stopping_conditions[condition_name][1]

    def evaluateStoppingConditions(self):
        """
        Check if one of the stopping conditions is satisfied
        :return: a boolean value indicating if a stopping condition is satisfied
        """

        for condition_name in self.__stopping_conditions.keys():
            stopping_condition, current_value = self.__stopping_conditions[condition_name]
            if condition_name == 'max_time':
                current_value = time.time() - current_value
            if stopping_condition.evaluateStoppingCondition(current_value):
                return True

        return False

    def outputData(self, f_list: np.array, **kwargs):
        """
        Call the Verbose_System instance to show the standard metrics and the ones specific for the algorithm at hand
        :param f_list: points in the objectives space
        :param kwargs: metrics specific for the algorithm at hand with related values

        Notes:  Argument verbose must be activated
        """

        if self._verbose:
            self.__verbose_system.outputData(self.__stopping_conditions['max_iter'][1],
                                             time.time() - self.__stopping_conditions['max_time'][1],
                                             self.__stopping_conditions['max_f_evals'][1],
                                             f_list,
                                             **kwargs)

    def showFigure(self, p_list: np.array, f_list: np.array, hold_still: bool = False):
        """
        Trigger the Graphical_Plot instance such that the Pareto front plot and the Pareto solutions one are shown
        :param p_list: Pareto solutions
        :param f_list: Pareto front points
        :param hold_still: if set to True, the execution is stopped; once that the plot window is closed, the execution continues

        Notes:  Argument plot_pareto_front must be activated.
                In order also to plot the Pareto solutions whenever is possible, plot_pareto_solutions must be activated.
                For more details, see parser_management.py.
        """

        if self._plot_pareto_front:
            self.__graphical_plot.showFigure(p_list, f_list, hold_still)

    def closeFigure(self):
        """
        Through the Graphical_Plot instance, close the figure of the plots
        """

        if self._plot_pareto_front:
            self.__graphical_plot.closeFigure()

