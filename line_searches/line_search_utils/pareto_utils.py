import numpy as np


def exists_sufficiently_dominating_point(new_f: np.array, f_list: np.array, alpha: float, beta: float, theta: float):
    """
    It indicates if the solution new_f is sufficiently dominated by another one contained in the array f_list
    :param new_f: solution values in the objectives space
    :param f_list: array of solutions
    :param alpha: the current value for the step size
    :param beta: the coefficient for the sufficient decrease condition
    :param theta: the optimal value of the search direction problem given the point from which the line search is started (for instance, see fmopg.py)
    :return: a boolean value indicating if the solution new_f is sufficiently dominated or not
    """

    if np.isnan(new_f).any():
        return True

    n_obj = len(new_f)

    suff_decr = np.reshape(new_f - beta * alpha * theta, (1, n_obj))
    dominance_matrix = f_list - suff_decr
    is_suff_dominated = (np.sum(dominance_matrix <= 0, axis=1) == n_obj).any()

    return is_suff_dominated
