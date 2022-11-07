import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable printing of tensorflow information and warnings.

import numpy as np
from datetime import datetime
import tensorflow as tf

from Algorithms.Algorithm_Utils.Graphical_Plot import Graphical_Plot
from Algorithms.Algorithm_Factory import Algorithm_Factory

from General_Utils.Args_Utils import printParameters, argsPreprocessing, argsFileCreation
from General_Utils.Management_Utils import folderInitialization, executionTimeFileInitialization, writeInExecutionTimeFile, writeResultsInCsvFile, savePlots
from General_Utils.Pareto_Utils import pointsInitialization, pointsPostprocessing
from General_Utils.Progress_Bar import Progress_Bar

from constants import PROBLEM_DIMENSIONALITIES
from parser_management import getArgs


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    # Arguments management. #

    args = getArgs()

    printParameters(args)
    algorithms_names, problems, n_problems, seeds, general_settings, algorithms_settings, DDS_settings, ALS_settings = argsPreprocessing(args)

    print('N° Algorithms: ', len(algorithms_names))
    print('N° Problems: ', n_problems)
    print('N° Seeds: ', len(seeds))
    print()

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if general_settings['verbose']:

        # Progress bar instantiation. #

        progress_bar = Progress_Bar(len(algorithms_names) * n_problems * len(seeds))
        progress_bar.showBar()

    for seed in seeds:

        print()
        print('Seed: ', seed)

        if general_settings['general_export']:

            # Creation of the folders to save the results, along with the file containing the arguments values and the files that will contain algorithms execution times (see the documentation of the following three functions). #

            folderInitialization(seed, date, algorithms_names)
            argsFileCreation(seed, date, args)
            executionTimeFileInitialization(seed, date, algorithms_names)

        for algorithm_name in algorithms_names:
            print('Algorithm: ', algorithm_name)

            for problem in problems:
                print('Problem: ', problem.name())

                # The problem dimensionalities to test are retrieved from PROBLEM_DIMENSIONALITIES (see constants.py). #

                var_range = PROBLEM_DIMENSIONALITIES[problem.familyName()]

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

                        # Retrieve the initial points from which the algorithm execution has to start. #

                        initial_p_list, initial_f_list, n_initial_points = pointsInitialization(problem_instance, 'hyper', n)

                        # Instantiation of the algorithm to test. #

                        algorithm = Algorithm_Factory.get_algorithm(algorithm_name,
                                                                    general_settings=general_settings,
                                                                    algorithms_settings=algorithms_settings,
                                                                    DDS_settings=DDS_settings,
                                                                    ALS_settings=ALS_settings)

                        problem_instance.evaluateFunctions(initial_p_list[0, :])
                        problem_instance.evaluateFunctionsJacobian(initial_p_list[0, :])

                        # Execution of the algorithm starting from the given initial points. #

                        p_list, f_list, elapsed_time = algorithm.search(initial_p_list, initial_f_list, problem_instance)

                        # Post-processing of the obtained points. #

                        filtered_p_list, filtered_f_list = pointsPostprocessing(p_list, f_list, problem_instance)

                        if general_settings['plot_pareto_front'] or general_settings['only_end_plot_pareto_front']:

                            # Plotting of the Pareto front and of the Pareto solutions (if requested and possible). #

                            graphical_plot = Graphical_Plot(bool(general_settings['plot_pareto_front'] * general_settings['plot_pareto_solutions']
                                                                 + general_settings['plot_pareto_front'] * general_settings['only_end_plot_pareto_solutions']
                                                                 + general_settings['only_end_plot_pareto_front'] * general_settings['only_end_plot_pareto_solutions']),
                                                            general_settings['plot_dpi'])
                            graphical_plot.showFigure(filtered_p_list, filtered_f_list, hold_still=True)
                            graphical_plot.closeFigure()

                        if general_settings['general_export']:

                            # Save Pareto front and Pareto solutions (if requested) in CSV files, Pareto front and Pareto solutions (if requested and possible) plots and execution time (see the documentation of the following three functions). #

                            writeInExecutionTimeFile(seed, date, algorithm_name, problem, n, elapsed_time)
                            writeResultsInCsvFile(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem, export_pareto_solutions=general_settings['general_export_pareto_solutions'])
                            savePlots(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem, general_settings['general_export_pareto_solutions'], general_settings['plot_dpi'])

                        if general_settings['verbose']:

                            # Progress bar update. #

                            progress_bar.incrementCurrentValue()
                            progress_bar.showBar()

                        tf.compat.v1.reset_default_graph()
                        session.close()
