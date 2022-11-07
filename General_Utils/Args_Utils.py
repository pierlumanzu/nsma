import os

from constants import PROBLEMS, PROBLEM_DIMENSIONALITIES

# For more details on the arguments, see parser_management.py.


def printParameters(args):
    """
    Print the values of the arguments
    :param args: the arguments and their values
    """

    if args.verbose:
        print()
        print('Parameters')
        print()

        for key in args.__dict__.keys():
            print(key.ljust(args.verbose_interspace), args.__dict__[key])
        print()


def checkArgs(args):
    """
    Check the validity of the arguments values
    :param args: the arguments and their values

    Notes:  It raises an AssertionError if one of the arguments has an invalid value.
    """

    assert len(args.seeds) > 0
    for seed in args.seeds:
        assert seed > 0

    if args.max_iter is not None:
        assert args.max_iter > 0
    if args.max_time is not None:
        assert args.max_time > 0
    if args.max_f_evals is not None:
        assert args.max_f_evals > 0

    assert args.verbose_interspace >= 1
    assert args.plot_dpi >= 1

    ####################################################
    ### NSMA ###
    ####################################################

    assert args.NM_pop_size > 0
    assert 0 <= args.NM_crossover_probability <= 1
    assert args.NM_crossover_eta >= 0
    assert args.NM_mutation_eta >= 0
    assert args.NM_shift > 0
    assert 0 <= args.NM_crowding_quantile <= 1
    assert args.NM_n_opt > 0
    assert args.NM_FMOPG_max_iter > 0
    assert args.NM_theta_for_stationarity <= 0
    assert args.NM_theta_tol <= 0
    assert args.NM_theta_tol < args.NM_theta_for_stationarity
    assert 0 < args.NM_theta_dec_factor < 1

    ####################################################
    ### NSGA-II ###
    ####################################################

    assert args.NG2_pop_size > 0
    assert 0 <= args.NG2_crossover_probability <= 1
    assert args.NG2_crossover_eta >= 0
    assert args.NG2_mutation_eta >= 0

    ####################################################
    ### Gurobi ###
    ####################################################

    assert -1 <= args.gurobi_method <= 5

    ####################################################
    ### Armijo-Type Line Search ###
    ####################################################

    assert args.ALS_alpha_0 > 0
    assert 0 < args.ALS_delta < 1
    assert 0 < args.ALS_beta < 1
    assert args.ALS_min_alpha > 0


def argsPreprocessing(args):
    """
    Check the arguments values with the function checkArgs(Object) and group them based on their semantic
    :param args: the arguments and their values
    :return: the list of algorithms names, the list of problems, the number of problems, the list of seeds, the general arguments, the algorithms arguments (each algorithm has its own arguments group), the gurobi-related arguments, the arguments of the Armijo-Type Line Search
    """

    checkArgs(args)

    algorithms_names = args.algorithms

    problems = []
    n_problems = 0

    for prob in args.problems:

        # The problems of the family to consider are retrieved by PROBLEMS (see constants.py). #

        problems.extend(PROBLEMS[prob])
        for problem in PROBLEMS[prob]:
            n_problems += len(PROBLEM_DIMENSIONALITIES[problem.familyName()])

    seeds = args.seeds

    general_settings = {'max_iter': args.max_iter,
                        'max_time': args.max_time,
                        'max_f_evals': args.max_f_evals,
                        'verbose': args.verbose,
                        'verbose_interspace': args.verbose_interspace,
                        'plot_pareto_front': args.plot_pareto_front,
                        'plot_pareto_solutions': args.plot_pareto_solutions,
                        'only_end_plot_pareto_front': args.only_end_plot_pareto_front,
                        'only_end_plot_pareto_solutions': args.only_end_plot_pareto_solutions,
                        'general_export': args.general_export,
                        'general_export_pareto_solutions': args.general_export_pareto_solutions,
                        'plot_dpi': args.plot_dpi}

    NM_settings = {'pop_size': args.NM_pop_size,
                   'crossover_probability': args.NM_crossover_probability,
                   'crossover_eta': args.NM_crossover_eta,
                   'mutation_eta': args.NM_mutation_eta,
                   'shift': args.NM_shift,
                   'crowding_quantile': args.NM_crowding_quantile,
                   'n_opt': args.NM_n_opt,
                   'FMOPG_max_iter': args.NM_FMOPG_max_iter,
                   'theta_for_stationarity': args.NM_theta_for_stationarity,
                   'theta_tol': args.NM_theta_tol,
                   'theta_dec_factor': args.NM_theta_dec_factor}

    NG2_settings = {'pop_size': args.NG2_pop_size,
                    'crossover_probability': args.NG2_crossover_probability,
                    'crossover_eta': args.NG2_crossover_eta,
                    'mutation_eta': args.NG2_mutation_eta}

    algorithms_settings = {'NSMA': NM_settings,
                           'NSGA-II': NG2_settings}

    DDS_settings = {'gurobi': args.gurobi,
                    'gurobi_method': args.gurobi_method,
                    'gurobi_verbose': args.gurobi_verbose}

    ALS_settings = {'alpha_0': args.ALS_alpha_0,
                    'delta': args.ALS_delta,
                    'beta': args.ALS_beta,
                    'min_alpha': args.ALS_min_alpha}

    return algorithms_names, problems, n_problems, seeds, general_settings, algorithms_settings, DDS_settings, ALS_settings


def argsFileCreation(seed: int, date: str, args):
    """
    Create and save a file where the arguments names and values are written
    :param seed: seed for the pseudo-random number generator
    :param date: date in which the execution has started
    :param args: the arguments and their values

    Notes:  The argument general_export must be activated.
    """

    if args.general_export:
        args_file = open(os.path.join('Execution_Outputs', date, str(seed), 'params.csv'), 'w')
        for key in args.__dict__.keys():
            if type(args.__dict__[key]) == float:
                args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
            else:
                args_file.write('{};{}\n'.format(key, args.__dict__[key]))
        args_file.close()


