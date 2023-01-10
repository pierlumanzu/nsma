import numpy as np

from algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm

from problems.problem import Problem


class FMOPG(GradientBasedAlgorithm):
    """
    The Front Multi-Objective Projected Gradient algorithm class

    The main functions are:
        - Initialize a FMOPG instance;
        - Execute the algorithm starting from a point of a given array;
        - Reset the stopping conditions current values.
    """

    def __init__(self,
                 theta_tol: float,
                 gurobi: bool,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 ALS_alpha_0: float,
                 ALS_delta: float,
                 ALS_beta: float,
                 ALS_min_alpha: float,
                 max_iter: int = None,
                 max_time: float = None,
                 max_f_evals: int = None):
        """
        Initialize a FMOPG instance
        :param theta_tol: it indicates the tolerance after which a point is considered Pareto-stationary; it can be seen as the epsilon value for the epsilon-Pareto-stationarity; for more details, the user is referred to the article
        :param gurobi: if set to True, the Gurobi Optimizer is used to solve the search direction problem
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        :param ALS_alpha_0: it indicates the initial step size for the Armijo-Type Line Search
        :param ALS_delta: it indicates the coefficient for the step size contraction
        :param ALS_beta: it indicates the coefficient for the sufficient decrease condition
        :param ALS_min_alpha: it indicates the minimum value of alpha that is considered by the Armijo-Type Line Search; after that, the line search fails returning a null step size
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes
        :param max_f_evals: maximum number of function evaluations

        Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
                For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
        """

        GradientBasedAlgorithm.__init__(self,
                                        max_iter, max_time, max_f_evals, False, 0, False, False, 0,
                                        theta_tol,
                                        gurobi, gurobi_method, gurobi_verbose, ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
                                        name_DDS='Boundconstrained_Projected_Gradient_DDS', name_ALS='BoundconstrainedFrontALS')

        self.__theta_array = np.array([-np.inf], dtype=float)
        GradientBasedAlgorithm.add_stopping_condition(self, 'theta_tolerance', theta_tol, self.__theta_array[0], equal_required=True)

        # An additional stopping condition regards the current step size. If it is 0, it will not be possible to optimize the point anymore and, then, the FMOPG algorithm execution can be stopped. #
        self.__alpha_array = np.array([1], dtype=float)
        GradientBasedAlgorithm.add_stopping_condition(self, 'min_alpha', 0, self.__alpha_array[0], smaller_value_required=True, equal_required=True)

    def search(self, p_list: np.array, f_list: np.array, problem: Problem, index_initial_point: int = None, I: tuple = None):
        """
        Execute the algorithm starting from a point of a given array
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param problem: the considered problem
        :param index_initial_point: the index of the problem solution to optimize
        :param I: the subset of objective functions indices to consider (see nsma.py)
        :return: the new arrays p_list and f_list; an array which contains for each processed point the optimal value of the search direction problem at that point (theta_array)

        Notes:  The index of the point to optimize can change during the iterations.
                For the stopping conditions 'theta_tolerance', only the last value of theta, i.e., the one related to the last point, is considered.
                For the stopping conditions 'min_alpha', only the last value of alpha, i.e., the one related to the last point, is considered.
        """

        n, m = p_list.shape[1], f_list.shape[1]
        index_point = index_initial_point

        while not self.evaluate_stopping_conditions():

            n_iteration = self.get_stopping_condition_current_value('max_iter')

            J = problem.evaluate_functions_jacobian(p_list[index_point, :])
            self.add_to_stopping_condition_current_value('max_f_evals', n)

            if self.evaluate_stopping_conditions():
                break

            # Solving the search direction problem. #
            # theta is the optimal value of the search direction problem at the current point; it indicates if the point is near to the Pareto-stationarity or not. #

            v, theta = self._direction_solver.compute_direction(problem, J[I,], x_p=p_list[index_point, :])
            self.__theta_array[n_iteration] = theta
            self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

            if theta < self._theta_tol:

                # Execution of the Armijo-Type Line Search. #

                new_p, new_f, alpha, f_eval_ls = self._line_search.search(problem, p_list[index_point, :], f_list, v, theta, np.array(list(I)))
                self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                self.__alpha_array[n_iteration] = alpha
                self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration])

                if new_p is not None:

                    # The new solution is added at the end of the p_list array. The related point in the objectives space is added in the f_list array in the same way. #

                    p_list = np.concatenate((p_list, new_p.reshape((1, n))), axis=0)
                    f_list = np.concatenate((f_list, new_f.reshape((1, m))), axis=0)

                    # The new point to optimize is in the last position of the p_list array. #

                    index_point = p_list.shape[0] - 1

                    self.__theta_array = np.concatenate((self.__theta_array, np.array([-np.inf])), axis=0)

                    # For the stopping condition 'theta_tolerance', only the last theta value is considered. #

                    self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration + 1])

                    self.__alpha_array = np.concatenate((self.__alpha_array, np.array([1])), axis=0)

                    # For the stopping condition 'min_alpha', only the last alpha value is considered. #

                    self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration + 1])

            self.add_to_stopping_condition_current_value('max_iter', 1)

        return p_list, f_list, self.__theta_array

    def reset_stopping_conditions_current_values(self, theta_tol: float):
        """
        Reset the stopping conditions current values
        :param theta_tol: the new current value for the stopping condition 'theta_tolerance'

        Notes:  The current values of the stopping conditions 'max_time' and 'max_f_evals' are changed by the memetic algorithm that employs FMOPG.
        """

        self.update_stopping_condition_current_value('max_iter', 0)

        self._theta_tol = theta_tol
        self.__theta_array = np.array([-np.inf], dtype=float)
        self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[0])
        self.update_stopping_condition_reference_value('theta_tolerance', theta_tol)

        self.__alpha_array = np.array([1], dtype=float)
        self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[0])
