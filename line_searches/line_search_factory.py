from line_searches.armijo_type.boundconstrained_front_als import BoundconstrainedFrontALS


class LineSearchFactory:
    """
    Factory to create line search instances

    Available line searches:
        - Bound-constrained Front Armijo-Type Line Search
    """

    @staticmethod
    def get_line_search(line_search_name: str, alpha_0: float, delta: float, beta: float, min_alpha: float):
        """
        Return an instance of a line search
        :param line_search_name: name of line search
        :param alpha_0: it indicates the initial step size for the line search
        :param delta: it indicates the coefficient for the step size contraction
        :param beta: it indicates the coefficient for the sufficient decrease condition
        :param min_alpha: it indicates the minimum value of alpha that is considered by the line search; after that, this latter one fails returning a null step size
        :return: the instance of the requested line search

        Notes:  It raises a NotImplementedError if the requested line search is not among the available ones.
        """

        if line_search_name == 'BoundconstrainedFrontALS':
            return BoundconstrainedFrontALS(alpha_0, delta, beta, min_alpha)

        else:
            raise NotImplementedError
