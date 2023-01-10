import numpy as np
from gurobipy import Model, GRB

from direction_solvers.descent_direction.dds import DDS
from direction_solvers.gurobi_settings import GurobiSettings

from problems.problem import Problem


class BoundconstrainedProjectedGradientDDSGurobiVersion(DDS, GurobiSettings):
    """
    Class to find the Bound-constrained Projected Gradient Descent Direction (Gurobi Version)

    The main functions are:
        - Initialize a class instance to solve the search direction problem;
        - Find the search direction at a specific solution for the problem at hand.

    Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
            For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
    """

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        """
        Initialize a class instance to solve the search direction problem
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        """

        DDS.__init__(self)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: Problem, Jac: np.array, x_p: np.array = None):
        """
        Find the search direction at a specific solution for the problem at hand
        :param problem: the considered problem
        :param Jac: the Jacobian matrix related to the objective functions calculated at x_p
        :param x_p: the problem solution
        :return: the Bound-constrained Projected Gradient Descent Direction

        Notes:  In this class, we implement the LP formulation reported in the paper, to which the user is referred for more details.
                The function raises an AssertionError if the problem solution x_p is not defined.
                The function handles the AttributeError: in this case, no descent direction is found and a null direction is returned; the value of theta, which is supposed to be the optimal value of the problem, is set equal to 0.
        """

        assert x_p is not None

        m, n = Jac.shape

        # If the Jacobian matrix is composed by some components that are infinity or Not A Number, a null direction and a null theta are returned. #

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model("Bound-constrained Projected Gradient Direction")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            # Variables settings. #

            z = model.addVars(n, lb=problem.lb, ub=problem.ub, name="z")
            beta = model.addVar(lb=-np.inf, ub=0., name="beta")

            # Objective function definition. #

            obj = beta
            model.setObjective(obj)

            # Constraints definition. #

            for j in range(m):
                linear_expression = sum(Jac[j, i] * (z[i] - x_p[i]) for i in range(n)) - beta
                model.addConstr(linear_expression <= 0)

            for i in range(n):
                linear_expression_upper = z[i] - x_p[i] - 1
                model.addConstr(linear_expression_upper <= 0)

                linear_expression_lower = - z[i] + x_p[i] - 1
                model.addConstr(linear_expression_lower <= 0)

            model.update()

            # Initial values for the variables. #

            for i in range(n):
                z[i].start = float(x_p[i])
            beta.start = 0.

            # Optimization of the problem. #

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()

                # The descent direction. #

                d_p = np.array([s.x for s in sol][:n]) - x_p

                # The optimal value of the problem; if it is close to 0, it means that the considered problem solution is near to the Pareto-stationarity. #

                theta_p = sol[-1].x
            else:

                # If the optimality is not reached, a null direction and a null theta are returned. #

                return np.zeros(n), 0

        except AttributeError:
            return np.zeros(n), 0

        return d_p, theta_p
