import numpy as np

from Algorithms.Genetic.Genetic_Utils.General_Utils import nonDominatedSorting, calcCrowdingDistance, splitByFeasibility

"""
Copyright (C) 2013-2017 Docker, Inc. - Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""


class Genetic_Survival_Strategy:
    """
    Class for the NSGA-II survival strategy (see NSGA.py).

    The main functions are:
        - Initialize a survival strategy instance;
        - Use the infeasible solutions survival strategy;
        - Use the feasible solutions survival strategy;
        - Execute the survival strategy.
    """

    def __init__(self, pop_size: int):
        """
        Initialize a survival strategy instance
        :param pop_size: size of the population
        """

        self._pop_size = pop_size

    def leastInfeasibleSurvival(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, n_survive: int):
        """
        Use the infeasible solutions survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param n_survive: maximum number of survivals
        :return: p_list, f_list and constraint_violations with only n_survive items

        Notes:  The solutions are sorted based on their constraint violations.
                If the number of solutions is greater than n_survive, then some solutions are dropped: in particular, the ones with the highest constraint violations are removed.
        """

        sorted_by_cv = constraint_violations[:, 0].argsort()
        p_list, f_list, constraint_violations = p_list[sorted_by_cv], f_list[sorted_by_cv], constraint_violations[sorted_by_cv]

        return p_list[:n_survive], f_list[:n_survive], constraint_violations[:n_survive]

    def rankingAndCrowdingFeasibleSurvival(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, n_survive: int):
        """
        Use the feasible solutions survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param n_survive: maximum number of survivals
        :return: p_list, f_list, constraint_violations, rank_list and crowding_list with only n_survive items

        Notes:  The solutions are sorted based, first, on their ranks, and, then, on their crowding distances
                If the number of solutions is greater than n_survive, then some solutions are dropped: in particular, the ones with the highest ranks or, in case of equal rank, with the lowest crowding distances are removed
        """

        survivors = []
        rank_list = np.array([[None]] * len(p_list))
        crowding_list = np.array([[None]] * len(p_list))

        fronts = nonDominatedSorting(f_list, n_survive)

        for k, front in enumerate(fronts):
            crowding_of_front = calcCrowdingDistance(f_list[front, :])

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

        return p_list[survivors], f_list[survivors], constraint_violations[survivors], rank_list[survivors].astype(int), crowding_list[survivors].astype(float)

    def getSurvivals(self, p_list: np.array, f_list: np.array, constraint_violations: np.array):
        """
        Execute the survival strategy
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :return: p_list, f_list, constraint_violations, rank_list and crowding_list

        Notes:  It divides the solutions in feasible and infeasible ones through the function splitByFeasibility(np.array)
                If there are no feasible solutions, then the function leastInfeasibleSurvival(np.array, np.array, np.array, int) is called to get the survivals among the infeasible points
                Otherwise, the function rankingAndCrowdingFeasibleSurvival(np.array, np.array, np.array, int) is first called to get the survivals among the feasible solutions
                Then, the function leastInfeasibleSurvival(np.array, np.array, np.array, int) is called to get the last survivals among the infeasible points
                The number of maximum survivals is chosen based on the size of p_list and the population size
                For each survived infeasible solution, the associated rank and crowding distance are equal to infinity
        """

        n_survive = min(self._pop_size, len(p_list))

        feasible, infeasible = splitByFeasibility(constraint_violations)

        p_list_f, f_list_f, constraint_violations_f, rank_list_f, crowding_list_f = None, None, None, None, None
        p_list_inf, f_list_inf, constraint_violations_inf = None, None, None

        if len(feasible) == 0:
            p_list_inf, f_list_inf, constraint_violations_inf = self.leastInfeasibleSurvival(p_list[infeasible], f_list[infeasible], constraint_violations[infeasible], min(len(infeasible), n_survive))
        else:
            p_list_f, f_list_f, constraint_violations_f, rank_list_f, crowding_list_f = self.rankingAndCrowdingFeasibleSurvival(p_list[feasible], f_list[feasible], constraint_violations[feasible],  min(len(feasible), n_survive))

            n_infeasible_survive = n_survive - len(p_list_f)

            if n_infeasible_survive > 0:
                p_list_inf, f_list_inf, constraint_violations_inf = self.leastInfeasibleSurvival(p_list[infeasible], f_list[infeasible], constraint_violations[infeasible], min(len(infeasible), n_infeasible_survive))

        if p_list_f is not None and p_list_inf is not None:
            p_list_s = np.concatenate((p_list_f, p_list_inf), axis=0)
            f_list_s = np.concatenate((f_list_f, f_list_inf), axis=0)
            constraint_violations_s = np.concatenate((constraint_violations_f, constraint_violations_inf), axis=0)
            rank_list_s = np.concatenate((rank_list_f, np.array([[np.inf]] * len(infeasible))), axis=0)
            crowding_list_s = np.concatenate((crowding_list_f, np.array([[np.inf]] * len(infeasible))), axis=0)
        elif p_list_f is not None:
            p_list_s = p_list_f
            f_list_s = f_list_f
            constraint_violations_s = constraint_violations_f
            rank_list_s = rank_list_f
            crowding_list_s = crowding_list_f
        else:
            p_list_s = p_list_inf
            f_list_s = f_list_inf
            constraint_violations_s = constraint_violations_inf
            rank_list_s = np.array([[np.inf]] * len(infeasible))
            crowding_list_s = np.array([[np.inf]] * len(infeasible))

        return p_list_s, f_list_s, constraint_violations_s, rank_list_s, crowding_list_s
