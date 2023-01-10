import numpy as np

from algorithms.genetic.genetic_utils.general_utils import non_dominated_sorting, calc_crowding_distance, split_by_feasibility
from algorithms.genetic.genetic_utils.survival_strategy import GeneticSurvivalStrategy


class MemeticSurvivalStrategy(GeneticSurvivalStrategy):
    """
    This class implements the survival strategy of NSMA. It is very similar to the one of NSGA-II (see GeneticSurvivalStrategy.py).
    However, it has additional features that are useful for the NSMA execution.

    The main functions are:
        - Initialize a survival strategy instance;
        - Use the infeasible solutions survival strategy;
        - Use the feasible solutions survival strategy;
        - Execute the survival strategy.
    """

    def __init__(self, pop_size: int, crowding_quantile: float):
        """
        Initialize a survival strategy instance
        :param pop_size: size of the population
        :param crowding_quantile: requested quantile of the finite crowding distances related to the rank-0 feasible points
        """

        GeneticSurvivalStrategy.__init__(self, pop_size)

        self.__crowding_quantile = crowding_quantile

    def least_infeasible_survival(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, n_survive: int, eps_pareto_stationarity: np.array = None):
        """
        Use the infeasible solutions survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param n_survive: maximum number of survivals
        :param eps_pareto_stationarity: an array indicating, for each problem solution and for each subset of the objective functions, if the solution is epsilon-Pareto-stationary w.r.t. that subset
        :return: p_list, f_list, constraint_violations and eps_pareto_stationarity with only n_survive items

        Notes:  The solutions are sorted based on their constraint violations.
                If the number of solutions is greater than n_survive, then some solutions are dropped: in particular, the ones with the highest constraint violations are removed.
                The function raises an AssertionError if the parameter eps_pareto_stationarity is not defined.
                The epsilon value for the epsilon-Pareto-stationarity is a NSMA parameter (theta_for_stationarity; see nsma.py).
        """

        assert eps_pareto_stationarity is not None

        sorted_by_cv = constraint_violations[:, 0].argsort()
        p_list, f_list, constraint_violations, eps_pareto_stationarity = p_list[sorted_by_cv], f_list[sorted_by_cv], constraint_violations[sorted_by_cv], eps_pareto_stationarity[sorted_by_cv]

        return p_list[:n_survive], f_list[:n_survive], constraint_violations[:n_survive], eps_pareto_stationarity[:n_survive]

    def ranking_and_crowding_feasible_survival(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, n_survive: int, eps_pareto_stationarity: np.array = None):
        """
        Use the feasible solutions survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param n_survive: maximum number of survivals
        :param eps_pareto_stationarity: an array indicating, for each problem solution and for each subset of the objective functions, if the solution is epsilon-Pareto-stationary w.r.t. that subset
        :return: p_list, f_list, constraint_violations, rank_list, crowding_list, eps_pareto_stationarity with only n_survive items; the quantile of the finite crowding distances related to the rank-0 feasible points (threshold_crowding_distance)

        Notes:  The solutions are sorted based, first, on their ranks, and, then, on their crowding distances
                If the number of solutions is greater than n_survive, then some solutions are dropped: in particular, the ones with the highest ranks or, in case of equal rank, with the lowest crowding distances are removed
                The function raises an AssertionError if the parameter eps_pareto_stationarity is not defined
                The epsilon value for the epsilon-Pareto-stationarity is a NSMA parameter (theta_for_stationarity; see nsma.py).
                If there are no rank-0 points with a finite crowding distance, then the threshold_crowding_distance is set to infinity.
        """

        assert eps_pareto_stationarity is not None

        survivors = []
        rank_list = np.array([[None]] * len(p_list))
        crowding_list = np.array([[None]] * len(p_list))

        fronts = non_dominated_sorting(f_list, n_survive)

        threshold_crowding_distance = np.inf

        for k, front in enumerate(fronts):
            crowding_of_front = calc_crowding_distance(f_list[front, :])

            if k == 0:
                filtered_crowding_of_front = crowding_of_front[np.isfinite(crowding_of_front)]
                if len(filtered_crowding_of_front) > 0:
                    threshold_crowding_distance = np.quantile(filtered_crowding_of_front, q=self.__crowding_quantile)

            for j, i in enumerate(front):
                rank_list[i] = k
                crowding_list[i] = crowding_of_front[j]

            if len(survivors) + len(front) > n_survive:
                P = np.random.permutation(len(crowding_of_front))
                I = np.argsort(crowding_of_front[P], kind='quicksort')
                I = P[I]
                I = np.flip(I, axis=0)
                I = I[:(n_survive - len(survivors))]
            else:
                I = np.arange(len(front))

            survivors.extend(front[I])

        return p_list[survivors], f_list[survivors], constraint_violations[survivors], rank_list[survivors].astype(int), crowding_list[survivors].astype(float), eps_pareto_stationarity[survivors], threshold_crowding_distance

    def get_survivals(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, eps_pareto_stationarity: np.array = None):
        """
        Execute the survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param eps_pareto_stationarity: an array indicating, for each problem solution and for each subset of the objective functions, if the solution is epsilon-Pareto-stationary w.r.t. that subset
        :return: p_list, f_list, constraint_violations, rank_list, crowding_list, eps_pareto_stationarity and the quantile of the finite crowding distances related to the rank-0 feasible points (threshold_crowding_distance)

        Notes:  It divides the solutions in feasible and infeasible ones through the function split_by_feasibility(np.array)
                If there are no feasible solutions, then the function least_infeasible_survival(np.array, np.array, np.array, int, np.array) is called to get the survivals among the infeasible points
                Otherwise, the function ranking_and_crowding_feasible_survival(np.array, np.array, np.array, int, np.array) is first called to get the survivals among the feasible solutions
                Then, the function least_infeasible_survival(np.array, np.array, np.array, int, np.array) is called to get the last survivals among the infeasible points
                The number of maximum survivals is chosen based on the size of p_list and the population size
                For each survived infeasible solution, the associated rank and crowding distance are equal to infinity
                The function raises an AssertionError if the parameter eps_pareto_stationarity is not defined
                The epsilon value for the epsilon-Pareto-stationarity is a NSMA parameter (theta_for_stationarity; see nsma.py).
                If the function ranking_and_crowding_feasible_survival(np.array, np.array, np.array, int, np.array) is not called, the threshold_crowding_distance variable remains undefined
        """

        assert eps_pareto_stationarity is not None

        n_survive = min(self._pop_size, len(p_list))

        feasible, infeasible = split_by_feasibility(constraint_violations)

        p_list_f, f_list_f, constraint_violations_f, rank_list_f, crowding_list_f, eps_pareto_stationarity_f = None, None, None, None, None, None
        p_list_inf, f_list_inf, constraint_violations_inf, eps_pareto_stationarity_inf = None, None, None, None
        threshold_crowding_distance = None

        if len(feasible) == 0:
            p_list_inf, f_list_inf, constraint_violations_inf, eps_pareto_stationarity_inf = self.least_infeasible_survival(p_list[infeasible], f_list[infeasible], constraint_violations[infeasible], min(len(infeasible), n_survive), eps_pareto_stationarity[infeasible])
        else:
            p_list_f, f_list_f, constraint_violations_f, rank_list_f, crowding_list_f, eps_pareto_stationarity_f, threshold_crowding_distance = self.ranking_and_crowding_feasible_survival(p_list[feasible], f_list[feasible], constraint_violations[feasible], min(len(feasible), n_survive), eps_pareto_stationarity[feasible])

            n_infeasible_survive = n_survive - len(p_list_f)

            if n_infeasible_survive > 0:
                p_list_inf, f_list_inf, constraint_violations_inf, eps_pareto_stationarity_inf = self.least_infeasible_survival(p_list[infeasible], f_list[infeasible], constraint_violations[infeasible], min(len(infeasible), n_infeasible_survive), eps_pareto_stationarity[infeasible])

        if p_list_f is not None and p_list_inf is not None:
            p_list_s = np.concatenate((p_list_f, p_list_inf), axis=0)
            f_list_s = np.concatenate((f_list_f, f_list_inf), axis=0)
            constraint_violations_s = np.concatenate((constraint_violations_f, constraint_violations_inf), axis=0)
            rank_list_s = np.concatenate((rank_list_f, np.array([[np.inf]] * len(infeasible))), axis=0)
            crowding_list_s = np.concatenate((crowding_list_f, np.array([[np.inf]] * len(infeasible))), axis=0)
            eps_pareto_stationarity_s = np.concatenate((eps_pareto_stationarity_f, eps_pareto_stationarity_inf), axis=0)
        elif p_list_f is not None:
            p_list_s = p_list_f
            f_list_s = f_list_f
            constraint_violations_s = constraint_violations_f
            rank_list_s = rank_list_f
            crowding_list_s = crowding_list_f
            eps_pareto_stationarity_s = eps_pareto_stationarity_f
        else:
            p_list_s = p_list_inf
            f_list_s = f_list_inf
            constraint_violations_s = constraint_violations_inf
            rank_list_s = np.array([[np.inf]] * len(infeasible))
            crowding_list_s = np.array([[np.inf]] * len(infeasible))
            eps_pareto_stationarity_s = eps_pareto_stationarity_inf

        return p_list_s, f_list_s, constraint_violations_s, rank_list_s, crowding_list_s, eps_pareto_stationarity_s, threshold_crowding_distance
