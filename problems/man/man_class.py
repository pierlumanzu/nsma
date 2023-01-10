from abc import ABC

from problems.problem import Problem

# The user is referred to the article for more details about the man problem. #


class MAN(Problem, ABC):
    """
    Abstract class for the man problem

    The main functions are:
        - Initialize a man problem instance;
        - Return the family name of the problem.
    """

    def __init__(self, n: int):
        """
        Initialize a man problem instance
        :param n: the problem dimensionality
        """

        Problem.__init__(self, n)

    @staticmethod
    def family_name():
        """
        Return the family name of the problem
        :return: the family name of the problem
        """

        return 'MAN'
