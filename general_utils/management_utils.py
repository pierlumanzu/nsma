import os
import numpy as np

from algorithms.algorithm_utils.graphical_plot import GraphicalPlot

from problems.problem import Problem


def make_folder(folder_path: str):
    """
    Make a specific directory if it does not already exist
    :param folder_path: the path of the directory to create
    """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def folder_initialization(seed: int, date: str, algorithms_names: list):
    """
    Initialize the folders where the computational results will be saved
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param algorithms_names: names of the tested algorithms

    Notes:  It raises an AssertionError if the folder "Execution_Outputs" does not exist: this one contains all the results of every execution.
            Inside the "Execution_Outputs" folder, a folder whose name is the date is created;
            Inside this latter one, a folder whose name is the value chosen for the seed is created.
            Inside the last folder, three folders are instantiated:
                - "Execution_Times", that contains for each Algorithm a file with the computational times that the Algorithm has reached with every tested problem;
                - "Csv", that contains a folder for each Algorithm where the Pareto front and the Pareto solutions, if requested, are written in CSV files;
                - "Plot", that contains a folder for each Algorithm where the plots of the Pareto fronts and of the Pareto solutions, if requested and possible (for more details, see parser_management.py), are saved.
            The function is called only if the argument general_export is activated.
    """

    assert os.path.exists(os.path.join('Execution_Outputs'))

    folders = ['Execution_Times', 'Csv', 'Plot']

    path = os.path.join('Execution_Outputs', date)
    make_folder(path)

    path = os.path.join(path, str(seed))
    make_folder(path)

    for index_folder, folder in enumerate(folders):
        make_folder(os.path.join(path, folder))
        if index_folder >= 1:
            for algorithm_name in algorithms_names:
                make_folder(os.path.join(path, folder, algorithm_name))


def execution_time_file_initialization(seed: int, date: str, algorithms_names: list):
    """
    Initialize a file for each tested Algorithm in the "Execution_Times" folder (see the function folder_initialization(int, str, list)).
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param algorithms_names: names of the tested algorithms

    Notes:  The function is called only if the argument general_export is activated.
    """

    for algorithm_name in algorithms_names:
        execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'w')
        execution_time_file.close()


def write_in_execution_time_file(seed: int, date: str, algorithm_name: str, problem: Problem, n: int, elapsed_time: float):
    """
    Write on the file in the "Execution_Times" folder related to the Algorithm at hand (see the function folder_initialization(int, str, list)) the computational time reached by the Algorithm on a problem
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param algorithm_name: name of the tested Algorithm
    :param problem: the considered problem
    :param n: the problem dimensionality
    :param elapsed_time: the computational time reached by the Algorithm on the problem

    Notes:  The function is called only if the argument general_export is activated.
    """

    execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'a')
    execution_time_file.write('Problem: ' + problem.__name__ + '    N: ' + str(n) + '    Time: ' + str(elapsed_time) + '\n')
    execution_time_file.close()


def write_results_in_csv_file(p_list: np.array, f_list: np.array, seed: int, date: str, algorithm_name: str, problem: Problem, export_pareto_solutions: bool = False):
    """
    Write in CSV files the Pareto front and the Pareto solutions, if requested, that an Algorithm has obtained on a problem (see the function folder_initialization(int, str, list))
    :param p_list: Pareto solutions
    :param f_list: Pareto front
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param algorithm_name: Name of the tested Algorithm
    :param problem: the considered problem
    :param export_pareto_solutions: if set to True, the Pareto solutions are saved in a CSV file (Notes: in order to be True, the argument general_export_pareto_solutions must be activated)

    Notes:  It raises an AssertionError if the arrays p_list and f_list do not contain the same number of points.
            The function is called only if the argument general_export is activated.
    """

    assert len(p_list) == len(f_list)
    n = p_list.shape[1]

    f_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_pareto_front.csv'.format(problem.__name__, n)), 'w')
    if len(f_list):
        for i in range(f_list.shape[0]):
            f_list_file.write(';'.join([str(round(el, 10)).replace('.', ',') for el in f_list[i, :]]) + '\n')
    f_list_file.close()

    if export_pareto_solutions:
        p_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_pareto_solutions.csv'.format(problem.__name__, n)), 'w')
        if len(p_list):
            for i in range(p_list.shape[0]):
                p_list_file.write(';'.join([str(round(el, 10)).replace('.', ',') for el in p_list[i, :]]) + '\n')
        p_list_file.close()


def save_plots(p_list: np.array, f_list: np.array, seed: int, date: str, algorithm_name: str, problem: Problem, export_pareto_solutions: bool, plot_dpi: int):
    """
    Save the plots of the Pareto front and of the Pareto solutions, if requested and possible (for more details, see parser_management.py), that an Algorithm has obtained on a problem (see the function folder_initialization(int, str, list))
    :param p_list: Pareto solutions
    :param f_list: Pareto front
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param algorithm_name: Name of the tested Algorithm
    :param problem: the considered problem
    :param export_pareto_solutions: if set to True, the plot of the Pareto solutions is saved whenever is possible (Notes: in order to be True, the argument general_export_pareto_solutions must be activated)
    :param plot_dpi: it indicates the dpi of the plot image(s) to save

    Notes:  It raises an AssertionError if the arrays p_list and f_list do not contain the same number of points.
            The function is called only if the argument general_export is activated.
    """

    assert len(p_list) == len(f_list)

    graphical_plot = GraphicalPlot(export_pareto_solutions, plot_dpi)
    graphical_plot.save_figure(p_list, f_list, os.path.join('Execution_Outputs', date, str(seed), 'Plot'), algorithm_name, problem.__name__)
