import time
import math
import numpy as np

from algorithms.genetic.genetic_algorithm import GeneticAlgorithm
from algorithms.genetic.genetic_utils.general_utils import binary_tournament, set_to_bounds_if_outside, remove_lists_duplicates_point

from problems.problem import Problem

"""
Copyright (C) 2013-2017 Docker, Inc. - Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""


class NSGAII(GeneticAlgorithm):
    """
    Class for the NSGA-II algorithm

    The main functions are:
        - Initialize a NSGA-II instance;
        - Execute the algorithm starting from a set of initial points of the problem at hand;
        - Get the solution parents;
        - Execute the crossover operation;
        - Execute the mutation operation;
        - Get the solution offsprings.

    Notes:  NSGA-II does not require any selection operator.
    """

    def __init__(self,
                 max_iter: int,
                 max_time: float,
                 max_f_evals: int,
                 verbose: bool,
                 verbose_interspace: int,
                 plot_pareto_front: bool,
                 plot_pareto_solutions: bool,
                 plot_dpi: int,
                 pop_size: int,
                 crossover_probability: float,
                 crossover_eta: float,
                 mutation_eta: float):
        """
        Initialize a NSGA-II instance
        :param max_iter: maximum number of iterations
        :param max_time: maximum number of elapsed minutes on a problem
        :param max_f_evals: maximum number of function evaluations
        :param verbose: if set to True, then the VerboseSystem instance is used during the algorithm iterations
        :param verbose_interspace: space between a metric and another one in the printing of the VerboseSystem instance
        :param plot_pareto_front: if set to True, the Pareto front is plotted at each algorithm iteration (see GraphicalPlot.py)
        :param plot_pareto_solutions: if set to True and whenever is possible, the Pareto solutions are plotted at each algorithm iteration
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        :param pop_size: size of the population
        :param crossover_probability: see the function crossover(np.array, np.array, Problem, np.array, np.array)
        :param crossover_eta: see the function crossover(np.array, np.array, Problem, np.array, np.array)
        :param mutation_eta: see the function mutation(np.array, Problem, np.array, np.array)
        """

        GeneticAlgorithm.__init__(self,
                                  max_iter,
                                  max_time,
                                  max_f_evals,
                                  verbose,
                                  verbose_interspace,
                                  plot_pareto_front,
                                  plot_pareto_solutions,
                                  plot_dpi,
                                  pop_size)

        self.__crossover_probability = crossover_probability
        self.__crossover_eta = crossover_eta
        self.__mutation_eta = mutation_eta

    def search(self, p_list: np.array, f_list: np.array, problem: Problem):
        """
        Execute the Algorithm starting from a set of initial points of the problem at hand
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param problem: the considered problem
        :return: the new solutions (p_list, f_list) and the elapsed time
        """

        self.update_stopping_condition_current_value('max_time', time.time())

        # Computation of the rank, crowding distance and constraint violation for each problem solution. #

        constraint_violations = np.array([[sum([constraint if constraint > 0 else 0 for constraint in problem.evaluate_constraints(p)])] for p in p_list])
        p_list, f_list, constraint_violations, rank_list, crowding_list = self._survival_strategy.get_survivals(p_list, f_list, constraint_violations)

        self.show_figure(p_list, f_list)

        while not self.evaluate_stopping_conditions():

            # VerboseSystem: the metrics specific for NSGA-II are the maximum finite rank, the maximum rank (including the infinity values) and the minimum crowding distance. #

            self.output_data(f_list, max_finite_rank=max(rank_list[np.isfinite(rank_list)]), max_rank=max(rank_list.flatten()), min_crowding_dist=min(crowding_list.flatten()))

            # Get the offsprings from the current population. #

            p_list_off = self.get_offsprings(p_list, f_list, constraint_violations, crowding_list, problem)

            if len(p_list_off) != 0:
                f_list_off = np.zeros((p_list_off.shape[0], problem.m))
                for index_p_off in range(p_list_off.shape[0]):
                    f_list_off[index_p_off, :] = problem.evaluate_functions(p_list_off[index_p_off, :])
                self.add_to_stopping_condition_current_value('max_f_evals', p_list_off.shape[0])

                constraint_violations_off = np.array([[sum([constraint if constraint > 0 else 0 for constraint in problem.evaluate_constraints(p_off)])] for p_off in p_list_off])

                p_list = np.concatenate((p_list, p_list_off), axis=0)
                f_list = np.concatenate((f_list, f_list_off), axis=0)
                constraint_violations = np.concatenate((constraint_violations, constraint_violations_off), axis=0)

                # Given the current population and the offsprings, the survival strategy is used to get at most pop_size survivals. #

                p_list, f_list, constraint_violations, rank_list, crowding_list = self._survival_strategy.get_survivals(p_list, f_list, constraint_violations)

            self.show_figure(p_list, f_list)

            self.add_to_stopping_condition_current_value('max_iter', 1)

        self.close_figure()

        self.output_data(f_list, max_finite_rank=max(rank_list[np.isfinite(rank_list)]), max_rank=max(rank_list.flatten()), min_crowding_dist=min(crowding_list.flatten()))

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')

    def get_parents(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, crowding_list: np.array, n_offspring: int = None):
        """
        Get the solution parents
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param crowding_list: an array containing all the crowding distances; each of them is related to a point in p_list
        :param n_offspring: number of offsprings to create
        :return: the parents couples

        Notes:  It raises an AssertionError if the parameter n_offspring is not defined.
                After randomly selecting the solution couples, it calls the function binary_tournament(np.array, np.array, np.array, np.array) to select a parent from each pair
                Then, each parent is associated to another one to form a couple from which the offsprings are created
        """

        assert n_offspring is not None

        n_select = math.ceil(n_offspring / 2)
        n_random = n_select * 4
        n_perms = math.ceil(n_random / len(p_list))

        perms = []
        for i in range(n_perms):
            perms.append(np.random.permutation(len(p_list)))
        P = np.concatenate(perms, axis=0)[:n_random]
        P = np.reshape(P, (n_select * 2, 2))

        S = binary_tournament(f_list, constraint_violations, crowding_list, P)

        return np.reshape(S, (n_select, 2))

    def calc_beta_q(self, beta: np.array, rand: np.array):
        """
        See the function crossover(np.array, np.array, Problem, np.array, np.array)
        """

        alpha = 2.0 - np.power(beta, -(self.__crossover_eta + 1.0))

        mask, mask_not = (rand <= (1.0 / alpha)), (rand > (1.0 / alpha))

        inner_beta_q = np.zeros(mask.shape)
        inner_beta_q[mask] = np.power((rand * alpha), (1.0 / (self.__crossover_eta + 1.0)))[mask]
        inner_beta_q[mask_not] = np.power((1.0 / (2.0 - rand * alpha)), (1.0 / (self.__crossover_eta + 1.0)))[mask_not]

        return inner_beta_q

    def inner_crossover(self, p_list_parents: np.array, problem: Problem, surrogate_lb: np.array = None, surrogate_ub: np.array = None):
        """
        See the function crossover(np.array, np.array, Problem, np.array, np.array)

        Notes:  It can handle the surrogate bounds instead of the original ones (see nsma.py).
        """

        _, n_matings, n_var = p_list_parents.shape

        x_l, x_u = surrogate_lb if surrogate_lb is not None else problem.filtered_lb, surrogate_ub if surrogate_ub is not None else problem.filtered_ub

        do_crossover = np.full(p_list_parents[0].shape, True)
        do_crossover[np.random.random((n_matings, n_var)) > 0.5] = False
        do_crossover[np.abs(p_list_parents[0] - p_list_parents[1]) <= 1.0e-14] = False

        y1 = np.min(p_list_parents, axis=0)
        y2 = np.max(p_list_parents, axis=0)

        rand = np.random.random((n_matings, n_var))

        delta = (y2 - y1)
        delta[delta < 1.0e-10] = 1.0e-10

        beta = 1.0 + (2.0 * (y1 - x_l) / delta)
        beta_q = self.calc_beta_q(beta, rand)
        c1 = 0.5 * ((y1 + y2) - beta_q * delta)

        beta = 1.0 + (2.0 * (x_u - y2) / delta)
        beta_q = self.calc_beta_q(beta, rand)
        c2 = 0.5 * ((y1 + y2) + beta_q * delta)

        b = np.random.random((n_matings, n_var)) <= 0.5
        val = np.copy(c1[b])
        c1[b] = c2[b]
        c2[b] = val

        c = np.copy(p_list_parents)

        c[0, do_crossover] = c1[do_crossover]
        c[1, do_crossover] = c2[do_crossover]

        c[0] = set_to_bounds_if_outside(c[0], x_l, x_u)
        c[1] = set_to_bounds_if_outside(c[1], x_l, x_u)

        return c

    def crossover(self, p_list: np.array = None, parents: np.array = None, problem: Problem = None, surrogate_lb: np.array = None, surrogate_ub: np.array = None):
        """
        Execute the crossover operation
        :param p_list: problem solutions
        :param parents: parent couples
        :param problem: the considered problem
        :param surrogate_lb: an array containing all the surrogate lower bounds; each of them is related to a solution component
        :param surrogate_ub: an array containing all the surrogate upper bounds; each of them is related to a solution component
        :return: the new solution offsprings

        Notes:  It can handle the surrogate bounds instead of the original ones (see nsma.py).
                It raises an AssertionError if one of the three first parameters (p_list, f_list, problem) is not defined
                For the theoretical aspects of the crossover operation, the user is referred to the NSGA-II original article:
                    K. Deb, A. Pratap, S. Agarwal and T. Meyarivan,
                    "A fast and elitist multiobjective genetic Algorithm: NSGA-II,"
                    in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2,
                    pp. 182-197, April 2002, doi: 10.1109/4235.996017
        """

        assert p_list is not None and parents is not None and problem is not None

        p_list_parents = p_list[parents.T].copy()
        do_crossover = np.random.random(len(parents)) < self.__crossover_probability

        _p_list_offsprings = self.inner_crossover(p_list_parents, problem, surrogate_lb, surrogate_ub)

        p_list_parents[:, do_crossover, :] = _p_list_offsprings[:, do_crossover, :]

        p_list_offsprings = p_list_parents

        p_list_offsprings = p_list_offsprings.reshape(-1, p_list_offsprings.shape[-1])

        return p_list_offsprings

    def mutation(self, off: np.array = None, problem: Problem = None, surrogate_lb: np.array = None, surrogate_ub: np.array = None):
        """
        Execute the mutation operation
        :param off: solution offsprings
        :param problem: the considered problem
        :param surrogate_lb: an array containing all the surrogate lower bounds; each of them is related to a solution component
        :param surrogate_ub: an array containing all the surrogate upper bounds; each of them is related to a solution component
        :return: the mutated solution offsprings

        Notes:  It can handle the surrogate bounds instead of the original ones (see nsma.py).
                It raises an AssertionError if one of the two first parameters (off, problem) is not defined
                For the theoretical aspects of the mutation operator, the user is referred to the NSGA-II original article:
                    K. Deb, A. Pratap, S. Agarwal and T. Meyarivan,
                    "A fast and elitist multiobjective genetic Algorithm: NSGA-II,"
                    in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2,
                    pp. 182-197, April 2002, doi: 10.1109/4235.996017
        """

        assert off is not None and problem is not None

        Y = np.full(off.shape, np.inf)
        n_var = off.shape[1]

        prob = 1.0 / n_var

        do_mutation = np.random.random(off.shape) < prob
        Y[:, :] = off

        x_l, x_u = surrogate_lb if surrogate_lb is not None else problem.filtered_lb, surrogate_ub if surrogate_ub is not None else problem.filtered_ub

        x_l = np.repeat(x_l[None, :], off.shape[0], axis=0)[do_mutation]
        x_u = np.repeat(x_u[None, :], off.shape[0], axis=0)[do_mutation]

        off = off[do_mutation]

        delta1 = (off - x_l) / (x_u - x_l)
        delta2 = (x_u - off) / (x_u - x_l)

        mut_pow = 1.0 / (self.__mutation_eta + 1.0)

        rand = np.random.random(off.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        delta_q = np.zeros(off.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.__mutation_eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        delta_q[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.__mutation_eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        delta_q[mask_not] = d[mask_not]

        _Y = off + delta_q * (x_u - x_l)

        _Y[_Y < x_l] = x_l[_Y < x_l]
        _Y[_Y > x_u] = x_u[_Y > x_u]

        Y[do_mutation] = _Y

        Y = set_to_bounds_if_outside(Y, surrogate_lb if surrogate_lb is not None else problem.filtered_lb, surrogate_ub if surrogate_ub is not None else problem.filtered_ub)

        return Y

    def selection(self):
        """
        The selection operator is not defined in the NSGA-II Algorithm
        """

        pass

    def get_offsprings(self, p_list: np.array, f_list: np.array, constraint_violations: np.array, crowding_list: np.array, problem: Problem, surrogate_lb: np.array = None, surrogate_ub: np.array = None):
        """
        Get the solution offsprings
        :param p_list: problem solutions
        :param f_list: related points in the objectives space
        :param constraint_violations: an array containing all the constraint violations; each of them is related to a point in p_list
        :param crowding_list: an array containing all the crowding distances; each of them is related to a point in p_list
        :param problem: the considered problem
        :param surrogate_lb: an array containing all the surrogate lower bounds; each of them is related to a solution component
        :param surrogate_ub: an array containing all the surrogate upper bounds; each of them is related to a solution component
        :return: the solution offsprings

        Notes:  It can handle the surrogate bounds instead of the original ones (see nsma.py).
                It calls the functions get_parents(np.array, np.array, np.array, np.array, int), crossover(np.array, np.array, Problem, np.array, np.array)
                and mutation(np.array, Problem, np.array, np.array) until pop_size different offsprings are generated or a maximum number of iterations is reached.
                In each iteration, the offsprings that are duplicates of other ones or of solutions in the current population are removed through the remove_lists_duplicates_point(np.array, Object) function.
        """

        off = []
        max_iterations = 100
        n_iterations = 0

        while len(off) < self._pop_size:
            n_remaining = self._pop_size - len(off)

            parents = self.get_parents(p_list, f_list, constraint_violations, crowding_list, n_remaining)
            _off = self.crossover(p_list, parents, problem, surrogate_lb, surrogate_ub)
            _off = self.mutation(_off, problem, surrogate_lb, surrogate_ub)
            _off = remove_lists_duplicates_point(_off, p_list, off)

            if len(off) + len(_off) > self._pop_size:
                n_remaining = self._pop_size - len(off)
                _off = _off[:n_remaining]

            off = _off if off == [] else np.concatenate((off, _off), axis=0)
            n_iterations += 1

            if n_iterations > max_iterations:
                break

        return off
