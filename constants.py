from problems.man.man_instance import MAN1

"""
Through the argument problems, the user can choose the family/families of problems (for more details, see parser_management.py).
Through PROBLEMS, the user can choose the problems of a family to evaluate with the indicated algorithms.
The problems can be provided for each family.
"""

PROBLEMS = {
    'MAN': [MAN1]
}

"""
Through PROBLEM_DIMENSIONALITIES, the user can choose the problem dimensionalities to test.
These latter ones can be indicated for all the problems of a family or for each problem individually.
In the second case, each problem class must contain an override of the function family_name() (for instance, see MOP_Instances.py).
"""

PROBLEM_DIMENSIONALITIES = {
    'MAN': [2, 5, 10, 20, 30, 40, 50, 100, 200]
}
