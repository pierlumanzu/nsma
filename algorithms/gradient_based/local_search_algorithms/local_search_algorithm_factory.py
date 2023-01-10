from algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG


class LocalSearchAlgorithmFactory:
    """
    Factory to create local search gradient-based algorithms instances

    Available algorithms:
        - FMOPG
    """

    @staticmethod
    def get_algorithm(algorithm_name: str, args_algorithm: dict):
        """
        Return a local search gradient-based algorithm instance
        :param algorithm_name: name of the Algorithm
        :param args_algorithm: arguments to take as inputs in the Algorithm constructor
        :return: an instance of the requested algorithm

        Notes:  It raises a NotImplementedError if the requested algorithm is not among the available ones.
        """

        if algorithm_name == 'FMOPG':

            local_search_algorithm = FMOPG(args_algorithm['theta_tol'],
                                           args_algorithm['gurobi'],
                                           args_algorithm['gurobi_method'],
                                           args_algorithm['gurobi_verbose'],
                                           args_algorithm['ALS_alpha_0'],
                                           args_algorithm['ALS_delta'],
                                           args_algorithm['ALS_beta'],
                                           args_algorithm['ALS_min_alpha'],
                                           args_algorithm['FMOPG_max_iter'] if args_algorithm['FMOPG_max_iter'] is not None else None,
                                           args_algorithm['max_time'] if args_algorithm['max_time'] is not None else None,
                                           args_algorithm['max_f_evals'] if args_algorithm['max_f_evals'] is not None else None)

        else:
            raise NotImplementedError

        return local_search_algorithm
