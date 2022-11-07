import numpy as np
from scipy.optimize import linprog

from Direction_Solvers.Descent_Direction.DDS import DDS

from Problems.Problem import Problem


class Boundconstrained_Projected_Gradient_DDS_ScipyVersion(DDS):
    """
    Class to find the Bound-constrained Projected Gradient Descent Direction (Scipy Version)

    The main functions are:
        - Initialize a class instance to solve the search direction problem;
        - Find the search direction at a specific solution for the problem at hand.

    Notes:  In the scipy version, we employ the HiGHS dual simplex solver, i.e., a wrapper of the C++ high performance dual revised simplex implementation (HSOL).
            For more details about this solver, see https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html#optimize-linprog-highs-ds.
            In addition, the user is referred to the following sites/papers:
                1- Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
                   “HiGHS - high performance software for linear optimization.”
                   Accessed 4/16/2020 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
                2- Huangfu, Q. and Hall, J. A. J.
                   “Parallelizing the dual revised simplex method.”
                   Mathematical Programming Computation, 10 (1),
                   119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    """

    def __init__(self):
        """
        Initialize a class instance to solve the search direction problem
        """

        DDS.__init__(self)

    def computeDirection(self, problem: Problem, Jac: np.array, x_p: np.array = None):
        """
        Find the search direction at a specific solution for the problem at hand
        :param problem: the considered problem
        :param Jac: the Jacobian matrix related to the objective functions calculated at x_p
        :param x_p: the problem solution
        :return: the Bound-constrained Projected Gradient Descent Direction

        Notes:  In this class, we implement the LP formulation reported in the paper, to which the user is referred for more details.
                The function raises an AssertionError if the problem solution x_p is not defined.
        """

        assert x_p is not None

        m, n = Jac.shape

        # If the Jacobian matrix is composed by some components that are infinity or Not A Number, a null direction and a null theta are returned. #

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        # Objective function definition. #

        c = np.array([0.] * n + [1.])

        # Constraints definition. #

        A_ub = np.column_stack((Jac, np.array([-1.] * len(Jac))))
        A_ub = np.concatenate((A_ub, np.column_stack((np.eye(n), [0.] * n))), axis=0)
        A_ub = np.concatenate((A_ub, np.column_stack((-np.eye(n), [0.] * n))), axis=0)

        b_ub = np.array([sum(Jac[j, i] * x_p[i] for i in range(n)) for j in range(m)] + [1 + x_p[i] for i in range(n)] + [1 - x_p[i] for i in range(n)])

        bounds = [(problem.lb[i], problem.ub[i]) for i in range(n)] + [(-np.inf, 0.)]

        # Optimization of the problem. #

        sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs-ds')

        if sol.status == 0:

            # The descent direction. #

            d_p = sol.x[:n] - x_p

            # The optimal value of the problem; if it is close to 0, it means that the considered problem solution is near to the Pareto-stationarity. #

            theta_p = sol.x[-1]
        else:

            # If the optimality is not reached, a null direction and a null theta are returned. #

            return np.zeros(n), 0

        return d_p, theta_p
