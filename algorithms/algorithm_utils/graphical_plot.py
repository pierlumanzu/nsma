import os.path

import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This import needs for three-dimensional plots. #


class GraphicalPlot:
    """
    Class for Pareto fronts and Pareto solutions plots.

    The main functions are:
        - Initialize a class instance;
        - Create a figure;
        - Initialize a single figure for the plots;
        - Initialize the requested figures for the plots;
        - Show the figure;
        - Close the figure;
        - Save the figure.

    Notes:  The class is designed only to handle Pareto fronts whose dimension is in {2, 3}.
            If the Pareto solutions dimension is not in {2, 3}, the plot is not shown.
    """

    def __init__(self, plot_pareto_solutions: bool, plot_dpi: int):
        """
        Initialize a class instance
        :param plot_pareto_solutions: it indicates if the Pareto solutions must be plotted or not
        :param plot_dpi: it indicates the dpi of the image(s) containing the plot(s)
        """

        # Command for computers with MacOSX operating system. #

        if platform.system() == 'Darwin':

            # Activation of the PyQt5 backend for the matplotlib library. #

            matplotlib.use('Qt5Agg')

        self.__fig = None
        self.__ax = None

        self.__plot_pareto_solutions = plot_pareto_solutions
        self.__plot_dpi = plot_dpi

    def figure_creation(self, figure_name: str, figure_size: tuple[float, float] = None):
        """
        Create a figure
        :param figure_name: name of the figure
        :param figure_size: size of the figure
        """

        self.__fig = plt.figure(figsize=figure_size) if figure_size is not None else plt.figure(figsize=(7.2, 4.8))
        self.__fig.canvas.manager.set_window_title('Plot figure')
        self.__fig.suptitle(figure_name, fontsize=15)

    def init_single_figure(self, dim: int, figure_name: str):
        """
        Initialize a single figure for the plots
        :param dim: plot dimension
        :param figure_name: name of the figure

        Notes:  It raises an AssertionError if the plot dimension is not in {2, 3}.
                It calls the function figure_creation(str, tuple[float, float]) to create the figure.
                It is called when either the Pareto front or the Pareto solutions must be plotted.
        """

        assert 1 < dim <= 3

        self.__fig = None
        self.__ax = None

        self.figure_creation(figure_name)

        if dim == 3:

            # Command for three-dimensional plots. #

            self.__ax = plt.axes(projection='3d')

    def init_main_figure(self, n: int, m: int):
        """
        Initialize the requested figures for the plots
        :param n: solution dimension
        :param m: front dimension

        Notes:  It raises an AssertionError if the front dimension is not in {2, 3}.
                If the solution dimension is not in {2, 3}, the figure for the related plot is not initialized.
                It calls:
                    - the function init_single_figure(int, str), if the Pareto solutions plot is not requested or not possible to show;
                    - the function figure_creation(str, tuple[float, float]), if the Pareto solutions plot is requested and its dimension is in {2, 3}.
                In the second case, two subplots, one for the Pareto solutions and one for the Pareto front, are created.
        """

        assert 1 < m <= 3

        self.__fig = None
        self.__ax = None

        if not self.__plot_pareto_solutions or n > 3 or n == 1:
            self.init_single_figure(m, 'Pareto front')
        else:
            self.figure_creation('', figure_size=(14.5, 4.8))
            self.__ax = [self.__fig.add_subplot(1, 2, 1, projection='3d' if n == 3 else None), self.__fig.add_subplot(1, 2, 2, projection='3d' if m == 3 else None)]
        
        self.__fig.show()
        self.__fig.canvas.draw()

    def show_figure(self, p_list: np.array, f_list: np.array, hold_still: bool = False):
        """
        Show the figure
        :param p_list: Pareto solutions
        :param f_list: Pareto front points
        :param hold_still: if set to True, the execution is stopped; once that the plot window is closed, the execution continues

        Notes:  It raises an AssertionError if:
                    - p_list and f_list do not have the same number of points;
                    - the front dimension is not in {2, 3}.
                If the solution dimension is not in {2, 3}, the related plot is not shown.
                If there are some points with some infinity or Not A Number components, the function manages an OverflowError not showing the plot.
        """
        assert len(p_list) == len(f_list)
        n, m = p_list.shape[1], f_list.shape[1]

        assert 1 < m <= 3

        if self.__fig is None:
            self.init_main_figure(n, m)

        if not self.__plot_pareto_solutions or n > 3 or n == 1:
            if m == 2:
                plt.cla()
                plt.plot(f_list[:, 0], f_list[:, 1], 'ro')

                plt.xlabel(r'$f_1$', fontsize=13)
                plt.ylabel(r'$f_2$', fontsize=13)
            else:
                self.__ax.cla()
                self.__ax.plot3D(f_list[:, 0], f_list[:, 1], f_list[:, 2], 'ro')

                self.__ax.set_xlabel(r'$f_1$', fontsize=13)
                self.__ax.set_ylabel(r'$f_2$', fontsize=13)
                self.__ax.set_zlabel(r'$f_3$', fontsize=13)
        else:
            self.__ax[0].cla()
            self.__ax[1].cla()
            self.__ax[0].set_title('Pareto solutions')
            self.__ax[1].set_title('Pareto front')

            if n == 2:
                self.__ax[0].plot(p_list[:, 0], p_list[:, 1], 'bo')
            else:
                self.__ax[0].plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], 'bo')

            self.__ax[0].set_xlabel(r'$x_1$', fontsize=13)
            self.__ax[0].set_ylabel(r'$x_2$', fontsize=13)
            if n == 3:
                self.__ax[0].set_zlabel(r'$x_3$', fontsize=13)

            if m == 2:
                self.__ax[1].plot(f_list[:, 0], f_list[:, 1], 'ro')
            else:
                self.__ax[1].plot3D(f_list[:, 0], f_list[:, 1], f_list[:, 2], 'ro')

            self.__ax[1].set_xlabel(r'$f_1$', fontsize=13)
            self.__ax[1].set_ylabel(r'$f_2$', fontsize=13)
            if m == 3:
                self.__ax[1].set_zlabel(r'$f_3$', fontsize=13)

        if hold_still:
            plt.show()
        else:
            try:
                self.__fig.canvas.draw()
                self.__fig.canvas.flush_events()
            except OverflowError:
                pass

    @staticmethod
    def close_figure():
        """
        Close the figure
        """

        plt.clf()
        plt.close()

    def save_figure(self, p_list: np.array, f_list: np.array, file_base_path: str, algorithm_name: str, problem_name: str):
        """
        Save the figure
        :param p_list: Pareto solutions
        :param f_list: Pareto front points
        :param file_base_path: the base file path
        :param algorithm_name: name of the Algorithm
        :param problem_name: name of the problem

        Notes:  In particular, it saves:
                    - the Pareto front plot, if the Pareto solutions are not requested or not possible to plot;
                    - the Pareto solutions and the Pareto front plots, in separated files, if the Pareto solutions are requested and their dimension is in {2, 3}.
                It raises an AssertionError if:
                    - p_list and f_list do not have the same number of points;
                    - the front dimension is not in {2, 3}.
        """
        assert len(p_list) == len(f_list)
        n, m = p_list.shape[1], f_list.shape[1]

        assert 1 < m <= 3

        if not self.__plot_pareto_solutions or n > 3 or n == 1:
            if self.__fig is None:
                self.init_single_figure(m, 'Pareto front')

            if m == 2:
                plt.cla()
                plt.plot(f_list[:, 0], f_list[:, 1], 'ro')

                plt.xlabel(r'$f_1$', fontsize=13)
                plt.ylabel(r'$f_2$', fontsize=13)
            else:
                self.__ax.cla()
                self.__ax.plot3D(f_list[:, 0], f_list[:, 1], f_list[:, 2], 'ro')

                self.__ax.set_xlabel(r'$f_1$', fontsize=13)
                self.__ax.set_ylabel(r'$f_2$', fontsize=13)
                self.__ax.set_zlabel(r'$f_3$', fontsize=13)

            self.__fig.savefig(os.path.join(file_base_path, algorithm_name, '{}_{}_pareto_front.pdf'.format(problem_name, n)), format='pdf', dpi=self.__plot_dpi)

            self.close_figure()

        else:
            self.init_single_figure(n, 'Pareto solutions')

            if n == 2:
                plt.plot(p_list[:, 0], p_list[:, 1], 'bo')

                plt.xlabel(r'$x_1$', fontsize=13)
                plt.ylabel(r'$x_2$', fontsize=13)
            else:
                self.__ax.plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], 'bo')

                self.__ax.set_xlabel(r'$x_1$', fontsize=13)
                self.__ax.set_ylabel(r'$x_2$', fontsize=13)
                self.__ax.set_zlabel(r'$x_3$', fontsize=13)

            self.__fig.savefig(os.path.join(file_base_path, algorithm_name, '{}_{}_pareto_solutions.pdf'.format(problem_name, n)), format='pdf', dpi=self.__plot_dpi)

            self.close_figure()

            self.init_single_figure(m, 'Pareto front')

            if m == 2:
                plt.plot(f_list[:, 0], f_list[:, 1], 'ro')

                plt.xlabel(r'$f_1$', fontsize=13)
                plt.ylabel(r'$f_2$', fontsize=13)
            else:
                self.__ax.plot3D(f_list[:, 0], f_list[:, 1], f_list[:, 2], 'ro')

                self.__ax.set_xlabel(r'$f_1$', fontsize=13)
                self.__ax.set_ylabel(r'$f_2$', fontsize=13)
                self.__ax.set_zlabel(r'$f_3$', fontsize=13)

            self.__fig.savefig(os.path.join(file_base_path, algorithm_name, '{}_{}_pareto_front.pdf'.format(problem_name, n)), format='pdf', dpi=self.__plot_dpi)

            self.close_figure()
