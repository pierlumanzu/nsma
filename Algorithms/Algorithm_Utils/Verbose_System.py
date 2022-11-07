import numpy as np

from General_Utils.Pareto_Utils import paretoEfficient


class Verbose_System:
    """
        Class for the verbose system.

        The main functions are:
            - Initialize a verbose system instance;
            - Print standard metrics and other ones specific for the algorithm at hand.
    """

    def __init__(self, interspace: int):
        """
        Initialize a verbose system instance
        :param interspace: printing space between two metrics

        Notes:  It raises an AssertionError if a non integer value is provided for interspace
        """

        assert type(interspace) == int

        self.__interspace = interspace
        self.__parameter_names = None

    def outputData(self, n_iter: int, elapsed_time: float, n_f_evals: int, f_list: np.array, **kwargs):
        """
        Print standard metrics and other ones specific for the algorithm at hand
        :param n_iter: number of iterations
        :param elapsed_time: elapsed time (seconds) on the problem at hand
        :param n_f_evals: number of function evaluations
        :param f_list: solutions values in the objectives space
        :param kwargs: metrics specific for the algorithm at hand with related values

        Notes:  Standard metrics: N° iteration, Elapsed time, N° of function evaluations, N° Non-dominated points, N° points, Ratio (%) between these last two numbers.
                First time, the name of the metrics specific for the algorithm at hand are saved.
                Then, the function raises an AssertionError if the specified metrics are not the same.
        """
        if self.__parameter_names is None:
            
            print()
            output_string = '|' + 'N°iter'.rjust(self.__interspace) + ' |' + 'Time(s)'.rjust(self.__interspace) + ' |' + 'N°f_evals'.rjust(self.__interspace) + ' ||' + 'ND-points'.rjust(self.__interspace) + ' |' + 'N°points'.rjust(self.__interspace) + ' |' + 'Ratio(%)'.rjust(self.__interspace) + ' ||'
            self.__parameter_names = []
            for key in kwargs.keys():
                self.__parameter_names.append(key)
                output_string += key.rjust(self.__interspace) + ' |'
            print(output_string)

        assert len(list(kwargs.keys())) == len(self.__parameter_names)
        for key in kwargs.keys():
            assert key in self.__parameter_names

        ND_points = np.count_nonzero(paretoEfficient(f_list))

        output_string = '|' + str(n_iter).rjust(self.__interspace) + ' |' + str(round(elapsed_time, 2)).rjust(self.__interspace) + ' |' + str(n_f_evals).rjust(self.__interspace) + ' ||' + str(ND_points).rjust(self.__interspace) + ' |' + str(len(f_list)).rjust(self.__interspace) + ' |' + str(round(ND_points / len(f_list) * 100, 2)).rjust(self.__interspace) + ' ||'
        for parameter_name in self.__parameter_names:
            output_string += str(round(kwargs[parameter_name], 2) if type(kwargs[parameter_name]) != str else kwargs[parameter_name]).rjust(self.__interspace) + ' |'
        print(output_string)
