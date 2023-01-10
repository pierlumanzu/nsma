import numpy as np
from typing import Union


class StoppingCondition:
    """
        Class for algorithm stopping conditions.

        The main functions are:
            - Initialize a stopping condition instance;
            - Evaluate the stopping condition;
            - Update the reference value.
    """

    def __init__(self, reference_value: float, smaller_value_required: bool = False, equal_required: bool = False):
        """
        Initialize a stopping condition instance
        :param reference_value: the reference value of the stopping condition
        :param smaller_value_required: if set to True, the current value(s) must be smaller than the reference value in order to stop the execution of the algorithm at hand
        :param equal_required: if set to True, the current value(s) can be also equal to the reference value in order to stop the execution of the algorithm at hand
        """

        self.__reference_value = reference_value
        self.__smaller_value_required = smaller_value_required
        self.__equal_required = equal_required

    def evaluate_stopping_condition(self, current_value: Union[float, int, np.float64, np.ndarray]):
        """
        Evaluate the stopping condition
        :param current_value: current value(s) to compare with the reference one
        :return: a boolean value indicating if the stopping condition is satisfied

        Notes:  It raises a NotImplementedError if current value is not of the accepted types.
                If multiple current values are provided, the stopping condition must be satisfied from each of them.
                If the stopping condition is satisfied, the algorithm at hand has to be stopped.
        """

        if self.__reference_value is not None:

            if type(current_value) in [float, int, np.float64]:
                if not self.__smaller_value_required:
                    if not self.__equal_required:
                        return current_value > self.__reference_value
                    else:
                        return current_value >= self.__reference_value
                else:
                    if not self.__equal_required:
                        return current_value < self.__reference_value
                    else:
                        return current_value <= self.__reference_value

            elif type(current_value) == np.ndarray:
                if not self.__smaller_value_required:
                    if not self.__equal_required:
                        return (current_value > self.__reference_value).all()
                    else:
                        return (current_value >= self.__reference_value).all()
                else:
                    if not self.__equal_required:
                        return (current_value < self.__reference_value).all()
                    else:
                        return (current_value <= self.__reference_value).all()

            else:
                raise NotImplementedError

        else:
            return False

    def update_reference_value(self, reference_value: float):
        """
        Update the reference value
        :param reference_value: the new reference value of the stopping condition
        """
        self.__reference_value = reference_value
