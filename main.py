import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable printing of tensorflow information and warnings.

import numpy as np
from datetime import datetime
import tensorflow as tf

from algorithms.algorithm_utils.graphical_plot import GraphicalPlot
from algorithms.algorithm_factory import AlgorithmFactory

from general_utils.args_utils import print_parameters, args_preprocessing, args_file_creation
from general_utils.management_utils import folder_initialization, execution_time_file_initialization, write_in_execution_time_file, write_results_in_csv_file, save_plots
from general_utils.pareto_utils import points_initialization, points_postprocessing
from general_utils.progress_bar import ProgressBarWrapper

from constants import PROBLEM_DIMENSIONALITIES
from parser_management import getArgs


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    # Arguments management. #

    args = getArgs()

    print_parameters(args)
    algorithms_names, problems, n_problems, seeds, general_settings, algorithms_settings, DDS_settings, ALS_settings = args_preprocessing(args)

    print('N° algorithms: ', len(algorithms_names))
    print('N° problems: ', n_problems)
    print('N° Seeds: ', len(seeds))
    print()

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if general_settings['verbose']:

        # Progress bar instantiation. #

        progress_bar = ProgressBarWrapper(len(algorithms_names) * n_problems * len(seeds))
        progress_bar.show_bar()

    for seed in seeds:

        print()
        print('Seed: ', seed)

        if general_settings['general_export']:

            # Creation of the folders to save the results, along with the file containing the arguments values and the files that will contain algorithms execution times (see the documentation of the following three functions). #

            folder_initialization(seed, date, algorithms_names)
            args_file_creation(seed, date, args)
            execution_time_file_initialization(seed, date, algorithms_names)

        for algorithm_name in algorithms_names:
            print('Algorithm: ', algorithm_name)

            for problem in problems:
                print('Problem: ', problem.name())

                # The problem dimensionalities to test are retrieved from PROBLEM_DIMENSIONALITIES (see constants.py). #

                var_range = PROBLEM_DIMENSIONALITIES[problem.family_name()]

                for n in var_range:
                    print()
                    print()
                    print('N: ', n)

                    session = tf.compat.v1.Session()
                    with session.as_default():

                        # Instantiation of the problem to evaluate. #

                        problem_instance = problem(n=n)

                        # Set the pseudo-random number generator with one of the seeds indicated by the user (see parser_management.py). #

                        np.random.seed(seed=seed)

                        # Retrieve the initial points from which the Algorithm execution has to start. #

                        initial_p_list, initial_f_list, n_initial_points = points_initialization(problem_instance, 'hyper', n)

                        # Instantiation of the Algorithm to test. #

                        algorithm = AlgorithmFactory.get_algorithm(algorithm_name,
                                                                   general_settings=general_settings,
                                                                   algorithms_settings=algorithms_settings,
                                                                   DDS_settings=DDS_settings,
                                                                   ALS_settings=ALS_settings)

                        problem_instance.evaluate_functions(initial_p_list[0, :])
                        problem_instance.evaluate_functions_jacobian(initial_p_list[0, :])

                        # Execution of the Algorithm starting from the given initial points. #

                        p_list, f_list, elapsed_time = algorithm.search(initial_p_list, initial_f_list, problem_instance)

                        # Post-processing of the obtained points. #

                        filtered_p_list, filtered_f_list = points_postprocessing(p_list, f_list, problem_instance)

                        if general_settings['plot_pareto_front'] or general_settings['only_end_plot_pareto_front']:

                            # Plotting of the Pareto front and of the Pareto solutions (if requested and possible). #

                            graphical_plot = GraphicalPlot(bool(general_settings['plot_pareto_front'] * general_settings['plot_pareto_solutions']
                                                                + general_settings['plot_pareto_front'] * general_settings['only_end_plot_pareto_solutions']
                                                                + general_settings['only_end_plot_pareto_front'] * general_settings['only_end_plot_pareto_solutions']),
                                                           general_settings['plot_dpi'])
                            graphical_plot.show_figure(filtered_p_list, filtered_f_list, hold_still=True)
                            graphical_plot.close_figure()

                        if general_settings['general_export']:

                            # Save Pareto front and Pareto solutions (if requested) in CSV files, Pareto front and Pareto solutions (if requested and possible) plots and execution time (see the documentation of the following three functions). #

                            write_in_execution_time_file(seed, date, algorithm_name, problem, n, elapsed_time)
                            write_results_in_csv_file(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem, export_pareto_solutions=general_settings['general_export_pareto_solutions'])
                            save_plots(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem, general_settings['general_export_pareto_solutions'], general_settings['plot_dpi'])

                        if general_settings['verbose']:

                            # Progress bar update. #

                            progress_bar.increment_current_value()
                            progress_bar.show_bar()

                        tf.compat.v1.reset_default_graph()
                        session.close()
