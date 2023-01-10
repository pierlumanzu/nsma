class GurobiSettings:
    """
    Class for the Gurobi settings

    The main function is:
        - Initialize a class instance for the Gurobi settings.

    Notes:  In order to use the Gurobi Optimizer, you need it installed in your computer and, in addition, you need a Gurobi Licence.
            For more details on Gurobi, the user is referred to the Gurobi website (https://www.gurobi.com/).
    """

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        """
        Initialize a class instance for the Gurobi settings
        :param gurobi_method: it indicates the method used by the Gurobi Optimizer
        :param gurobi_verbose: if set to True, it enables the verbosity for the Gurobi optimizer
        """

        self._gurobi_method = gurobi_method
        self._gurobi_verbose = gurobi_verbose
