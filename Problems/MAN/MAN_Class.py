from abc import ABC

from Problems.Problem import Problem

# The user is referred to the article for more details about the MAN problem. #


class MAN(Problem, ABC):
    """
    Abstract class for the MAN problem

    The main functions are:
        - Initialize a MAN problem instance;
        - Return the family name of the problem.
    """

    def __init__(self, n: int):
        """
        Initialize a MAN problem instance
        :param n: the problem dimensionality
        """

        Problem.__init__(self, n)

    @staticmethod
    def familyName():
        """
        Return the family name of the problem
        :return: the family name of the problem
        """

        return 'MAN'
