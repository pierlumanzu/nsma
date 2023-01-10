import scipy.spatial
import numpy as np

""" 
Copyright (C) 2013-2017 Docker, Inc. - Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""


def binary_tournament(f_list: np.array, constraint_violations: np.array, crowding_list: np.array, P: np.array):
    """
    Given an array of solutions pairs, the binary tournament is used to choose one solution for each couple; this one will be a parent (for instance, see NSGA.py)
    :param f_list: solutions values in the objectives space
    :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in f_list
    :param crowding_list: an array containing all the crowding distances; each of them is related to a point in f_list
    :param P: the array containing the solutions pairs
    :return: an array containing the parents to generate the offsprings

    Notes: In a pair, the solutions are compared based on the following criteria.
            1) Constraint violation: the solution with the lowest constraint violation is preferred;
            2) Domination: if a solution dominates the other, the first one is chosen;
            3) Crowding distance: the solution with the highest crowding distance is preferred;
            4) Random: one of the two solutions is randomly chosen
    """

    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        if constraint_violations[a] > 0.0 or constraint_violations[b] > 0.0:
            if constraint_violations[a] < constraint_violations[b]:
                S[i] = a
            elif constraint_violations[a] > constraint_violations[b]:
                S[i] = b
            else:
                S[i] = np.random.choice([a, b])
        else:
            val = 0
            for j in range(len(f_list[a])):
                if f_list[a, j] < f_list[b, j]:
                    if val == -1:
                        val = 0
                        break
                    val = 1
                elif f_list[b, j] < f_list[a, j]:
                    if val == 1:
                        val = 0
                        break
                    val = -1
            if val == 1:
                S[i] = a
            elif val == -1:
                S[i] = b

            if np.isnan(S[i]):
                if crowding_list[a] > crowding_list[b]:
                    S[i] = a
                elif crowding_list[a] < crowding_list[b]:
                    S[i] = b
                else:
                    S[i] = np.random.choice([a, b])

    return S[:, None].astype(np.int, copy=False)


def calc_domination_matrix(f_list: np.array):
    """
    Compute the domination matrix
    :param f_list: solutions values in the objectives space
    :return: the domination matrix

    Notes: In each matrix cell (i, j), the values can be:
            - 1: solution i dominates solution j
            - -1: solution j dominates solution i
    """
    n, m = f_list.shape[0], f_list.shape[0]

    L = np.repeat(f_list, m, axis=0)
    R = np.tile(f_list, (n, 1))

    smaller = np.reshape(np.any(L < R, axis=1), (n, m))
    larger = np.reshape(np.any(L > R, axis=1), (n, m))

    M = np.logical_and(smaller, np.logical_not(larger)) * 1 + np.logical_and(larger, np.logical_not(smaller)) * -1

    return M


def fast_non_dominated_sort(f_list: np.array):
    """
    Divide the solutions in fronts
    :param f_list: solutions values in the objectives space
    :return: the fronts

    Notes: It computes the domination matrix through the function calc_domination_matrix(np.array)
           Then, it associates each solution to a rank and to a front in the following way:
                - the first front (rank 0) is composed by the points not dominated by any other one;
                - the second front (rank 1) is composed by the points only dominated by the ones with rank 0 but not dominated by any other one;
                - and so on...
           This procedure is continued until all the points are assigned to a specific rank and front
    """

    M = calc_domination_matrix(f_list)

    n = M.shape[0]

    fronts = []
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)
    is_dominating = [[] for _ in range(n)]
    n_dominated = np.zeros(n, dtype=int)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):

            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    fronts.append(current_front)

    while n_ranked < n:

        next_front = []

        for i in current_front:

            for j in is_dominating[i]:

                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts


def non_dominated_sorting(f_list: np.array, n_stop_if_ranked: int):
    """
    Divide the solutions in fronts; Discard the extra fronts when a number of points greater than or equal to n_stop_if_ranked is reached
    :param f_list: solutions values in the objectives space
    :param n_stop_if_ranked: number of points after which the extra fronts are discarded
    :return: the resulting fronts

    Notes: It calls the function fast_non_dominated_sort(np.array) to associate each solution to a rank and a front.
           It sequentially considers the fronts, starting from the first one (rank 0), until a number of points greater than or equal to n_stop_if_ranked is reached
           Then, the remaining fronts are discarded and not returned
    """

    fronts = fast_non_dominated_sort(f_list)

    _fronts = []
    n_ranked = 0
    for front in fronts:

        _fronts.append(np.array(front, dtype=int))
        n_ranked += len(front)

        if n_ranked >= n_stop_if_ranked:
            break

    fronts = _fronts

    return fronts


def split_by_feasibility(constraint_violations: np.array):
    """
    Divide the solutions based on their feasibility
    :param constraint_violations: an array containing all the constraint violations; each of them is related to a solution
    :return: an array of feasible solutions, an array of infeasible solutions
    """

    b = (constraint_violations <= 0)

    feasible = np.where(b)[0]
    infeasible = np.where(np.logical_not(b))[0]

    infeasible = infeasible[np.argsort(constraint_violations[infeasible, 0])]

    return feasible, infeasible


def set_to_bounds_if_outside(X: np.array, xl: np.array, xu: np.array):
    """
    Project the given solutions in a bounding box
    :param X: problem solutions
    :param xl: an array containing the lower bounds; each of them is related to a solution component
    :param xu: an array containing the upper bounds; each of them is related to a solution component
    :return: the projected problem solutions
    """

    xl = np.repeat(xl[None, :], X.shape[0], axis=0)
    X[X < xl] = xl[X < xl]
    xu = np.repeat(xu[None, :], X.shape[0], axis=0)
    X[X > xu] = xu[X > xu]

    return X


def find_duplicates(p_list: np.array, epsilon: float = 1e-24):
    """
    Find the duplicates in an array
    :param p_list: problem solutions
    :param epsilon: tolerance value to indicate if a solution is a duplicate of another one
    :return: an array indicating for each problem solution if it is a duplicate or not
    """

    D = scipy.spatial.distance.cdist(p_list, p_list)
    D[np.triu_indices(len(p_list))] = np.inf
    is_duplicate = np.any(D < epsilon, axis=1)

    return is_duplicate


def calc_crowding_distance(f_list: np.array, filter_out_duplicates: bool = True):
    """
    Calculate the crowding distance for each solution
    :param f_list: solutions values in the objectives space
    :param filter_out_duplicates: if set to True, it does not consider the duplicates in the crowding distance computation; in order to find these, the function find_duplicates(np.array) is employed
    :return: an array containing all the crowding distances; each of them is related to a point in f_list

    Notes:  In order to understand how this distance is calculated, the user is referred to
                K. Deb, A. Pratap, S. Agarwal and T. Meyarivan,
                "A fast and elitist multiobjective genetic algorithm: NSGA-II,"
                in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2,
                pp. 182-197, April 2002, doi: 10.1109/4235.996017.
            If the number of points is smaller than or equal to 2, the crowding distance cannot be computed.
            In this case, a crowding distance equal to infinity is assigned to each point.
    """

    n_points, n_obj = f_list.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:
        if filter_out_duplicates:
            is_unique = np.where(np.logical_not(find_duplicates(f_list)))[0]
        else:
            is_unique = np.arange(n_points)

        _F = f_list[is_unique]

        I = np.argsort(_F, axis=0, kind='mergesort')

        _F = _F[I, np.arange(n_obj)]

        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    return crowding


def remove_duplicates_point(p_list_1: np.array, f_list_1: np.array = None, constraint_violations_1: np.array = None, p_list_2: np.array = None, epsilon: float = 1e-16):
    """
    If p_list_2 is undefined, remove the duplicate problem solutions in the array p_list_1; if p_list_2 is provided, it removes the items in p_list_1 that are duplicates of ones in p_list_2; if f_list_1 and constraint_violations_1 are also provided, it removes the duplicates-related items from these two arrays too
    :param p_list_1: problem solutions
    :param f_list_1: related points in the objective space
    :param constraint_violations_1: an array containing all the constraint violations; each of them is related to a point in p_list_1
    :param p_list_2: second array of problem solutions
    :param epsilon: tolerance value to indicate if a solution is a duplicate of another one
    :return: the array p_list_1 without the duplicates; if f_list_1 and constraint_violations_1 are provided, these latter ones are also returned without the duplicates-related items
    """

    is_duplicate = np.array([False] * p_list_1.shape[0])

    if p_list_2 is None:
        D = scipy.spatial.distance.cdist(p_list_1, p_list_1)
        D[np.triu_indices(len(p_list_1))] = np.inf
    else:
        D = scipy.spatial.distance.cdist(p_list_1, p_list_2)

    D[np.isnan(D)] = np.inf

    is_duplicate[np.any(D < epsilon, axis=1)] = True
    p_list_1 = p_list_1[~is_duplicate]

    if f_list_1 is not None and constraint_violations_1 is not None:
        f_list_1 = f_list_1[~is_duplicate]
        constraint_violations_1 = constraint_violations_1[~is_duplicate]

        return p_list_1, f_list_1, constraint_violations_1

    else:
        return p_list_1


def remove_lists_duplicates_point(p_list: np.array, *args):
    """
    If only p_list is provided, remove the duplicate solutions from it; if other arrays are passed as parameters through args, it removes each item in p_list that is a duplicate of one in these arrays
    :param p_list: problem solutions
    :param args: other arrays
    :return: the array p_list without the duplicate items
    """

    p_list = remove_duplicates_point(p_list)
    for arg in args:
        if len(arg) > 0:
            if len(p_list) == 0:
                break
            else:
                p_list = remove_duplicates_point(p_list, p_list_2=arg)

    return p_list
