from Algorithms.Genetic.NSGA_II import NSGA_II
from Algorithms.Memetic.NSMA import NSMA


class Algorithm_Factory:
    """
    Factory to create algorithm instances

    Available algorithms:
        - NSMA
        - NSGA-II
    """

    @staticmethod
    def get_algorithm(algorithm_name: str, **kwargs):
        """
        Return an algorithm instance
        :param algorithm_name: name of the algorithm
        :param kwargs: arguments to take as inputs in the algorithm constructor
        :return: an instance of the requested algorithm

        Notes:  It raises a NotImplementedError if the requested algorithm is not among the available ones.
        """

        general_settings = kwargs['general_settings']

        algorithms_settings = kwargs['algorithms_settings']

        if algorithm_name == 'NSMA':
            NM_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']

            algorithm = NSMA(general_settings['max_iter'],
                             general_settings['max_time'],
                             general_settings['max_f_evals'],
                             general_settings['verbose'],
                             general_settings['verbose_interspace'],
                             general_settings['plot_pareto_front'],
                             general_settings['plot_pareto_solutions'],
                             general_settings['plot_dpi'],
                             NM_settings['pop_size'],
                             NM_settings['crossover_probability'],
                             NM_settings['crossover_eta'],
                             NM_settings['mutation_eta'],
                             NM_settings['shift'],
                             NM_settings['crowding_quantile'],
                             NM_settings['n_opt'],
                             NM_settings['FMOPG_max_iter'],
                             NM_settings['theta_for_stationarity'],
                             NM_settings['theta_tol'],
                             NM_settings['theta_dec_factor'],
                             DDS_settings['gurobi'],
                             DDS_settings['gurobi_method'],
                             DDS_settings['gurobi_verbose'],
                             ALS_settings['alpha_0'],
                             ALS_settings['delta'],
                             ALS_settings['beta'],
                             ALS_settings['min_alpha'])

        elif algorithm_name == 'NSGA-II':
            NG2_settings = algorithms_settings[algorithm_name]

            algorithm = NSGA_II(general_settings['max_iter'],
                                general_settings['max_time'],
                                general_settings['max_f_evals'],
                                general_settings['verbose'],
                                general_settings['verbose_interspace'],
                                general_settings['plot_pareto_front'],
                                general_settings['plot_pareto_solutions'],
                                general_settings['plot_dpi'],
                                NG2_settings['pop_size'],
                                NG2_settings['crossover_probability'],
                                NG2_settings['crossover_eta'],
                                NG2_settings['mutation_eta'])

        else:
            raise NotImplementedError

        return algorithm
