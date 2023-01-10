from direction_solvers.descent_direction.boundconstrained_projected_gradient_dds_gurobi_version import BoundconstrainedProjectedGradientDDSGurobiVersion
from direction_solvers.descent_direction.boundconstrained_projected_gradient_dds_scipy_version import BoundconstrainedProjectedGradientDDSScipyVersion


class DirectionSolverFactory:
    """
    Factory to create instances to solve the search direction problem

    Available classes:
        - Bound-constrained Projected Gradient Descent Direction (Gurobi version)
        - Bound-constrained Projected Gradient Descent Direction (Scipy version)

    Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
            For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
            In the scipy version, we employ the HiGHS dual simplex solver, i.e., a wrapper of the C++ high performance dual revised simplex implementation (HSOL).
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

    @staticmethod
    def get_direction_calculator(direction_type: str, gurobi_version: bool, gurobi_method: int, gurobi_verbose: bool):
        """
        Return an instance to solve the search direction problem
        :param direction_type: the instance class
        :param gurobi_version: if set to True, the Gurobi Optimizer is employed
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        :return: an instance of the requested class to solve the search direction problem

        Notes:  It raises a NotImplementedError if the requested class is not among the available ones.
        """

        if gurobi_version:
            if direction_type == 'Boundconstrained_Projected_Gradient_DDS':
                return BoundconstrainedProjectedGradientDDSGurobiVersion(gurobi_method, gurobi_verbose)
            else:
                raise NotImplementedError
        
        else:
            if direction_type == 'Boundconstrained_Projected_Gradient_DDS':
                return BoundconstrainedProjectedGradientDDSScipyVersion()
            else:
                raise NotImplementedError
