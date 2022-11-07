from progressbar import ProgressBar


class Progress_Bar:
    """
    Class for the progress bar.
    It indicates the number of the tested problems, the number of all the problems to test and the elapsed time of the execution.

    The main functions are:
            - Create an instance of the progress bar;
            - Increment the value of the tested problems;
            - Show the progress bar instance.

    Notes: The argument verbose must be activated.
    """

    def __init__(self, max_value: int):
        """
        Create an instance of the progress bar
        :param max_value: the number of all the problems to test
        """

        self.__bar = ProgressBar(max_value=max_value)
        self.__current_value = 0

    def incrementCurrentValue(self):
        """
        Increment the value of the tested problems
        """

        self.__current_value += 1

    def showBar(self):
        """
        Show the progress bar instance
        """

        self.__bar.update(self.__current_value)
