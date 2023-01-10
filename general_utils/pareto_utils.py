import numpy as np

from problems.problem import Problem


def points_initialization(problem: Problem, mod: str, n_initial_points: int):
    """
    If possible, instantiate n_initial_points solutions related to the problem at hand
    :param problem: the considered problem
    :param mod: the points instantiation modality (see problem.py)
    :param n_initial_points: the requested number of initial points
    :return: the problem solutions (p_list), the related points in the objectives space (f_list), the effective number of initial points
    """

    p_list = problem.generate_feasible_points_array(mod, n_initial_points)

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    return p_list, f_list, n_initial_points


def points_postprocessing(p_list: np.array, f_list: np.array, problem: Problem):
    """
    Remove the infeasible and the dominated solutions found by an Algorithm on a problem
    :param p_list: the problem solutions
    :param f_list: the points in the objectives space, each of them related to a problem solution
    :param problem: the considered problem
    :return: the new arrays p_list and f_list after removing the infeasible and the dominated solutions

    Notes:  It raises an AssertionError if the arrays p_list and f_list do not contain the same number of points.
    """

    assert len(p_list) == len(f_list)
    n_points, n = p_list.shape

    for p in range(n_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    feasible = [True] * n_points
    infeasible_points = 0
    for p in range(n_points):
        constraints = problem.evaluate_constraints(p_list[p, :])
        if (constraints > 0).any():
            feasible[p] = False
            infeasible_points += 1
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    efficient_point_idx = pareto_efficient(f_list)
    p_list = p_list[efficient_point_idx, :]
    f_list = f_list[efficient_point_idx, :]

    print('Results: found {} points'.format(len(p_list)))
    print()

    return p_list, f_list


def pareto_efficient(f_list: np.array):
    """
    It indicates the mutually non-dominated points in the array f_list
    :param f_list: points in the objectives space
    :return: an array of boolean values indicating for each point in the array f_list if it is not dominated by any other one in the array (True) or it is (False)
    """

    n_points, m = f_list.shape
    efficient = np.array([False] * n_points, dtype=bool)

    _, index = np.unique(f_list, return_index=True, axis=0)
    index = sorted(index)
    duplicates = [el for el in np.arange(n_points) if el not in index]
    indices = np.arange(n_points)

    for i in range(n_points):
        partial_ix = duplicates + [i]
        partial_matrix = f_list[np.delete(indices, partial_ix), :]
        dominance_matrix = partial_matrix - np.reshape(f_list[i, :], newshape=(1, m))
        is_dominated = (np.sum(dominance_matrix <= 0, axis=1) == m).any()
        if not is_dominated:
            efficient[i] = True

    return efficient
