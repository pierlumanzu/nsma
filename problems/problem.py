import numpy as np
import tensorflow as tf
from abc import abstractmethod


class Problem:
    """
    Abstract class for problems

    The main functions are:
        - Initialize a problem instance;
        - Evaluate the objective functions given a point;
        - Compute the Jacobian matrix related to the objective functions given a point;
        - Evaluate all the constraints, including the bound ones, given a point;
        - Compute the Jacobian matrix related to the constraints, including the bound ones, given a point;
        - Check the feasibility of a given point;
        - Generate a feasible random point;
        - Generate an array containing feasible points;
        - Return the objectives instances (see the function objectives());
        - Set the objectives instances;
        - Return the constraints except the bound ones (see the function general_constraints());
        - Set the constraints except the bound ones;
        - Return the lower bounds;
        - Set the lower bounds;
        - Return the filtered lower bounds (see the Constructor and the function lb(np.array));
        - Return the upper bounds;
        - Set the upper bounds;
        - Return the filtered upper bounds (see the Constructor and the function ub(np.array));
        - Return the problem dimensionality;
        - Return the number of objective functions;
        - Return the total number of constraints, including the bound ones;
        - Return the number of constraints, excluding the bound ones;
        - Return the number of bound constraints;
        - Return the number of lower bound constraints;
        - Return the number of upper bound constraints;
        - Return the name of the problem (to be implemented);
        - Return the family name of the problem (to be implemented).
    """

    def __init__(self, n: int):
        """
        Initialize a problem instance
        :param n: the problem dimensionality
        """

        self.__n = n

        # Problem variables. #

        self._z = tf.compat.v1.placeholder(dtype=tf.double, shape=[n, ])

        # Objective functions and their gradient. #

        self.__objectives = np.empty(0)
        self.__objectives_gradients = np.empty(0)

        # Boolean value indicating if a problem is constrained or not. #

        self.__constrained = False

        # General constraints (bound ones not considered) and their gradient. #

        self.__general_constraints = np.empty(0)
        self.__general_constraints_gradients = np.empty(0)

        """
        Lower bounds
            - lb: the effective lower bounds;
            - filtered_lb: the lower bounds without infinity values, i.e., -np.inf -> -2.0e19; they are useful with operations where infinity values cannot be taken into account;
            - is_finite_lb: an array of boolean values indicating for each component of the array lb if it is a finite value or not.
        """

        self.__lb = np.array([-np.inf] * self.__n, dtype=float)
        self.__filtered_lb = np.array([-2.0e19] * self.__n, dtype=float)
        self.__is_finite_lb = np.array([False] * self.__n, dtype=bool)

        """
        Upper bounds
            - ub: the effective upper bounds;
            - filtered_ub: the upper bounds without infinity values, i.e., np.inf -> 2.0e19; they are useful with operations where infinity values cannot be taken into account;
            - is_finite_ub: an array of boolean values indicating for each component of the array ub if it is a finite value or not.
        """

        self.__ub = np.array([np.inf] * self.__n, dtype=float)
        self.__filtered_ub = np.array([2.0e19] * self.__n, dtype=float)
        self.__is_finite_ub = np.array([False] * self.__n, dtype=bool)

    def evaluate_functions(self, x: np.array):
        """
        Evaluate the objective functions given a point
        :param x: the point
        :return: the values of the objective functions
        """

        return np.array([obj.eval({self._z: x}) for obj in self.__objectives])

    def evaluate_functions_jacobian(self, x: np.array):
        """
        Compute the Jacobian matrix related to the objective functions given a point
        :param x: the point
        :return: the Jacobian matrix related to the objective functions
        """

        jacobian = np.zeros((self.m, self.__n))
        for i in range(self.m):
            if self.__objectives_gradients[i] is not None:
                jacobian[i, :] = self.__objectives_gradients[i].eval({self._z: x})

        return jacobian

    def evaluate_constraints(self, x: np.array):
        """
        Evaluate all the constraints, including the bound ones, given a point
        :param x: the point
        :return: the values of the constraints
        """

        if self.__constrained:
            assert self.n_total_constraints != 0
            constraints_evaluated = None

            if self.n_general_constraints != 0:
                constraints_evaluated = np.array([general_constr.eval({self._z: x}) for general_constr in self.__general_constraints])

            if self.n_lower_constraints != 0:
                constraints_evaluated = np.concatenate((constraints_evaluated, self.__lb[self.__is_finite_lb] - x[self.__is_finite_lb])) if constraints_evaluated is not None else self.__lb[self.__is_finite_lb] - x[self.__is_finite_lb]

            if self.n_upper_constraints != 0:
                constraints_evaluated = np.concatenate((constraints_evaluated, x[self.__is_finite_ub] - self.__ub[self.__is_finite_ub])) if constraints_evaluated is not None else x[self.__is_finite_ub] - self.__ub[self.__is_finite_ub]

            return constraints_evaluated

        return np.empty(0)

    def evaluate_constraints_jacobian(self, x: np.array):
        """
        Compute the Jacobian matrix related to the constraints, including the bound ones, given a point
        :param x: the point
        :return: the Jacobian matrix related to the constraints, including the bound ones
        """

        if self.__constrained:
            assert self.n_total_constraints != 0
            constraints_jacobian = np.empty((self.n_total_constraints, self.__n))

            if self.n_general_constraints != 0:
                for i in range(self.n_general_constraints):
                    if self.__general_constraints_gradients[i] is not None:
                        constraints_jacobian[i, :] = self.__general_constraints_gradients[i].eval({self._z: x})

            if self.n_lower_constraints != 0:
                constraints_jacobian[self.n_general_constraints: self.n_general_constraints + self.n_lower_constraints, :] = -np.identity(self.__n)[self.__is_finite_lb, :]

            if self.n_upper_constraints != 0:
                constraints_jacobian[self.n_general_constraints + self.n_lower_constraints: self.n_total_constraints, :] = np.identity(self.n)[self.__is_finite_ub, :]

            return constraints_jacobian

        return np.empty(0)

    def check_point_feasibility(self, x: np.array):
        """
        Check the feasibility of a given point
        :param x: the point
        :return: a boolean value indicating if the point is feasible or not
        """

        if self.__constrained:
            assert self.n_total_constraints != 0

            if len(self.__lb) != 0:
                if (self.__lb > x).any():
                    return False

            if len(self.__ub) != 0:
                if (self.__ub < x).any():
                    return False

            return (np.array([general_constr.eval({self._z: x}) for general_constr in self.__general_constraints]) <= 0).all() if self.n_general_constraints != 0 else True

        return True

    def generate_feasible_random_point(self):
        """
        Generate a feasible random point
        :return: the point
        """

        scale = self.__filtered_ub - self.__filtered_lb - 2e-3
        shift = self.__filtered_lb + 1e-3

        x = shift + np.random.rand(self.__n) * scale
        while not self.check_point_feasibility(x):
            x = shift + np.random.rand(self.__n) * scale

        return x

    def generate_feasible_points_array(self, mod: str, size: int):
        """
        Generate an array containing feasible points
        :param mod: the modality of the points instantiation
        :param size: the number of points to instantiate
        :return: the array containing the points

        Notes:  It raises an AssertionError if the number of requested points is not a positive number.
                It raises an AssertionError if the mod is set to neither 'rand' nor 'hyper'.
                    - 'rand': all the points are randomly instantiated.
                    - 'hyper': the points are chosen from the hyper-diagonal formed by the lower and upper bounds.
                If the mod is set to 'hyper', the function raises an AssertionError if the number of the constraints, excluding the bound ones, is different from 0.
                Indeed, if the general constraints are set, there is no guarantee that the points in the hyper-diagonal are feasible.
                If the mod is set to 'hyper' and the number of requested points is 1 (a smaller number is not possible, otherwise an AssertionError would be raised), the point in the middle of the hyper-diagonal is chosen.
        """

        assert size > 0
        assert mod in ['rand', 'hyper']

        if mod.lower() == 'rand':
            p_list = np.zeros((size, self.__n), dtype=float)
            for i in range(size):
                p_list[i, :] = self.generate_feasible_random_point()
        else:
            assert self.n_general_constraints == 0

            scale = self.__filtered_ub - self.__filtered_lb - 2e-3
            shift = self.__filtered_lb + 1e-3

            p_list = np.zeros((size, self.__n), dtype=float)
            for i in range(size):
                p_list[i, :] = shift + ((i / (size - 1)) if size > 1 else 0.5) * scale

        return p_list

    @property
    def objectives(self):
        """
        Return the objectives instances

        Notes:  For safety reasons, if this function is called, it raises a RuntimeError.
                The function only needs to implement the setter function for the objectives (see the function objectives(list)).
        """

        raise RuntimeError

    @objectives.setter
    def objectives(self, objectives: list):
        """
        Set the objectives instances
        :param objectives: the objectives instances

        Notes:  It raises an AssertionError if the user tries to set an objective that is Not A Number or infinity.
        """

        for obj in objectives:
            assert obj is not np.nan and obj is not np.inf and obj is not -np.inf
        self.__objectives = objectives
        self.__objectives_gradients = [tf.gradients(obj, self._z)[0] for obj in self.__objectives]

    @property
    def general_constraints(self):
        """
        Return the constraints except the bound ones

        Notes:  For safety reasons, if this function is called, it raises a RuntimeError.
                The function only needs to implement the setter function for the constraints except the bound ones (see the function general_constraints(list)).
        """

        raise RuntimeError

    @general_constraints.setter
    def general_constraints(self, general_constraints: list):
        """
        Set the constraints except the bound ones
        :param general_constraints: the constraints except the bound ones

        Notes:  It raises an AssertionError if the user tries to set a constraint that is Not A Number or infinity.
        """

        for general_constr in general_constraints:
            assert general_constr is not np.nan and general_constr is not np.inf and general_constr is not -np.inf

        # When a constraint is set by the user, the problem automatically becomes constrained. #

        self.__constrained = True

        self.__general_constraints = general_constraints
        self.__general_constraints_gradients = [tf.gradients(general_constr, self._z)[0] for general_constr in self.__general_constraints]

    @property
    def lb(self):
        """
        Return the lower bounds
        :return: the lower bounds
        """

        return self.__lb

    @lb.setter
    def lb(self, lb: np.array):
        """
        Set the lower bounds
        :param lb: the lower bounds

        Notes:  It raises an AssertionError if the length of the array lb is not exactly equal to the problem dimensionality.
                It raises an AssertionError if there are some components of lb that are Not A Number.
                It raises an AssertionError if there are some components of lb that are np.inf (-np.inf is allowed).
        """

        assert len(lb) == self.__n
        assert not np.isnan(np.sum(lb))
        assert (lb != np.inf).all()

        # When a constraint is set by the user, the problem automatically becomes constrained. #

        self.__constrained = True

        self.__lb = lb
        self.__is_finite_lb = np.isfinite(self.__lb)

        self.__filtered_lb = self.__lb
        self.__filtered_lb[~self.__is_finite_lb] = -2.0e19

    @property
    def filtered_lb(self):
        """
        Return the filtered lower bounds (see the Constructor and the function lb(np.array))
        :return: the filtered lower bounds
        """

        return self.__filtered_lb

    @property
    def ub(self):
        """
        Return the upper bounds
        :return: the upper bounds
        """

        return self.__ub

    @ub.setter
    def ub(self, ub: np.array):
        """
        Set the upper bounds
        :param ub: the upper bounds

        Notes:  It raises an AssertionError if the length of the array ub is not exactly equal to the problem dimensionality.
                It raises an AssertionError if there are some components of ub that are Not A Number.
                It raises an AssertionError if there are some components of ub that are -np.inf (np.inf is allowed).
        """

        assert len(ub) == self.__n
        assert not np.isnan(np.sum(ub))
        assert (ub != -np.inf).all()

        # When a constraint is set by the user, the problem automatically becomes constrained. #

        self.__constrained = True

        self.__ub = ub
        self.__is_finite_ub = np.isfinite(self.__ub)

        self.__filtered_ub = self.__ub
        self.__filtered_ub[~self.__is_finite_ub] = 2.0e19

    @property
    def filtered_ub(self):
        """
        Return the filtered upper bounds (see the Constructor and the function ub(np.array))
        :return: the filtered upper bounds
        """

        return self.__filtered_ub

    @property
    def n(self):
        """
        Return the problem dimensionality
        :return: the problem dimensionality
        """

        return self.__n

    @property
    def m(self):
        """
        Return the number of objective functions
        :return: the number of objective functions
        """

        return len(self.__objectives)

    @property
    def n_total_constraints(self):
        """
        Return the total number of constraints, including the bound ones
        :return: the total number of constraints, including the bound ones
        """

        return self.n_general_constraints + self.n_bound_constraints

    @property
    def n_general_constraints(self):
        """
        Return the number of constraints, excluding the bound ones
        :return: the number of constraints, excluding the bound ones
        """

        return len(self.__general_constraints) if self.__constrained else 0

    @property
    def n_bound_constraints(self):
        """
        Return the number of bound constraints
        :return: the number of bound constraints
        """

        return self.n_lower_constraints + self.n_upper_constraints

    @property
    def n_lower_constraints(self):
        """
        Return the number of lower bound constraints
        :return: the number of lower bound constraints
        """

        return np.count_nonzero(self.__is_finite_lb) if self.__constrained else 0

    @property
    def n_upper_constraints(self):
        """
        Return the number of upper bound constraints
        :return: the number of upper bound constraints
        """

        return np.count_nonzero(self.__is_finite_ub) if self.__constrained else 0

    @staticmethod
    @abstractmethod
    def name():
        """
        Return the name of the problem (to be implemented)
        :return: the name of the problem
        """

        return

    @staticmethod
    @abstractmethod
    def family_name():
        """
        Return the family name of the problem (to be implemented)
        :return: the family name of the problem
        """

        return
