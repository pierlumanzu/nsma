import argparse
import sys


def get_args():

    parser = argparse.ArgumentParser(description='memetic and genetic algorithms for Global Multi-Objective Optimization')

    parser.add_argument('-a', '--algorithms', type=str, help='algorithms to execute', nargs='+', choices=['NSMA', 'NSGA-II'])

    parser.add_argument('-p', '--problems', help='Family Problem to test', nargs='+', choices=['MAN'])

    parser.add_argument('-s', '--seeds', help='Seeds for the pseudo-random number generator', nargs='+', type=int)

    parser.add_argument('-mi', '--max_iter', help='Maximum number of iterations for an Algorithm on a problem', default=None, type=int)

    parser.add_argument('-mt', '--max_time', help='Maximum number of elapsed minutes for an Algorithm on a problem', default=None, type=float)

    parser.add_argument('-mfe', '--max_f_evals', help='Maximum number of function evaluations for an Algorithm on a problem', default=None, type=int)

    parser.add_argument('-v', '--verbose', help='Verbose during the algorithms iterations', action='store_true', default=False)

    parser.add_argument('-vi', '--verbose_interspace', help='Space between a metric and an another one in the Verbose System (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('-ppf', '--plot_pareto_front', help='Plot Pareto front during the algorithms iterations', action='store_true', default=False)

    parser.add_argument('-pps', '--plot_pareto_solutions', help='Plot Pareto solutions during the algorithms iterations (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('-oeppf', '--only_end_plot_pareto_front', help='Plot Pareto front only at the end of the Algorithm execution', action='store_true', default=False)

    parser.add_argument('-oepps', '--only_end_plot_pareto_solutions', help='Plot Pareto solutions only at the end of the Algorithm execution (Requirements: plot_pareto_front or only_end_plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('-ge', '--general_export', help='Save Pareto fronts in CSV files, Pareto fronts plots, execution times and arguments files', action='store_true', default=False)

    parser.add_argument('-geps', '--general_export_pareto_solutions', help='Save Pareto solutions in CSV files, Pareto solutions plots if possible, i.e.,  problem dimensionality in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('-pd', '--plot_dpi', help='Dpi of the saved plot images (Requirements: general_export activated)', default=100, type=int)

    ####################################################
    ### NSMA ###
    ####################################################

    # For more details, see nsma.py. Alternatively, the user is referred to the article. #

    parser.add_argument('-NM_ps', '--NM_pop_size', help='NSMA parameter -- Population size', default=100, type=int)

    parser.add_argument('-NM_cp', '--NM_crossover_probability', help='NSMA parameter -- Crossover probability', default=0.9, type=float)

    parser.add_argument('-NM_ce', '--NM_crossover_eta', help='NSMA parameter -- Crossover eta', default=20, type=float)

    parser.add_argument('-NM_me', '--NM_mutation_eta', help='NSMA parameter -- Mutation eta', default=20, type=float)

    parser.add_argument('-NM_s', '--NM_shift', help='NSMA parameter -- Shift parameter', default=10, type=float)

    parser.add_argument('-NM_cq', '--NM_crowding_quantile', help='NSMA parameter -- Crowding distance quantile', default=0.9, type=float)

    parser.add_argument('-NM_no', '--NM_n_opt', help='NSMA parameter -- Number of iterations before doing optimization', default=5, type=int)

    parser.add_argument('-NM_F_mi', '--NM_FMOPG_max_iter', help='NSMA parameter -- Number of maximum iterations for FMOPG', default=5, type=int)

    parser.add_argument('-NM_ts', '--NM_theta_for_stationarity', help='NSMA parameter -- Theta for Pareto stationarity', default=-1.0e-10, type=float)

    parser.add_argument('-NM_tt', '--NM_theta_tol', help='NSMA parameter -- Theta tolerance', default=-1.0e-1, type=float)

    parser.add_argument('-NM_tdf', '--NM_theta_dec_factor', help='NSMA parameter -- Theta decreasing factor', default=10 ** (-1 / 2), type=float)

    ####################################################
    ### NSGA-II ###
    ####################################################

    """
    For more details, see NSGA-II.py. 
    Alternatively, the user is referred to
        K. Deb, A. Pratap, S. Agarwal and T. Meyarivan,
        "A fast and elitist multiobjective genetic Algorithm: NSGA-II,"
        in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2,
        pp. 182-197, April 2002, doi: 10.1109/4235.996017.
    """

    parser.add_argument('-NG2_ps', '--NG2_pop_size', help='NSGA-II parameter -- Population size', default=100, type=int)

    parser.add_argument('-NG2_cp', '--NG2_crossover_probability', help='NSGA-II parameter -- Crossover probability', default=0.9, type=float)

    parser.add_argument('-NG2_ce', '--NG2_crossover_eta', help='NSGA-II parameter -- Crossover eta', default=20, type=float)

    parser.add_argument('-NG2_me', '--NG2_mutation_eta', help='NSGA-II parameter -- Mutation eta', default=20, type=float)

    ####################################################
    ### Gurobi ###
    ####################################################

    """
    In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
    For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
    """

    parser.add_argument('-g', '--gurobi', help='Use the Gurobi Optimizer', action='store_true', default=False)

    parser.add_argument('-gm', '--gurobi_method', help='Gurobi parameter -- Method used in the Gurobi Optimizer', default=1, type=int)

    parser.add_argument('-gv', '--gurobi_verbose', help='Gurobi parameter -- Enable verbosity for the Gurobi Optimizer', action='store_true', default=False)

    ####################################################
    ### Armijo-Type Line Search ###
    ####################################################

    # For more details, see als.py. #

    parser.add_argument('-ALS_a0', '--ALS_alpha_0', help='ALS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('-ALS_d', '--ALS_delta', help='ALS parameter -- Coefficient for the step size contraction', default=0.5, type=float)

    parser.add_argument('-ALS_b', '--ALS_beta', help='ALS parameter -- Coefficient for the sufficient decrease condition', default=1.0e-4, type=float)

    parser.add_argument('-ALS_ma', '--ALS_min_alpha', help='ALS parameter -- Minimum possible value for the step size', default=1.0e-7, type=float)

    return parser.parse_args(sys.argv[1:])

