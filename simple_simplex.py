#!/usr/bin/env python
from sys import stdin, stdout
from pprint import pprint
from StringIO import StringIO

import numpy as np
import sympy
from sympy.core.relational import LessThan

import logging
l = logging.getLogger()
l.setLevel(logging.DEBUG)

EPSILON = 1e-10

class LP_STATE(object):
    UNSOLVED = 0
    OPTIMAL = 1
    UNBOUNDED = 2
    UNFEASIBLE = 3

def get_variable_assignments(pivoted_tableau, basis):
    """
    :param pivoted_tableau: Tableau on which `simplex` was already performed.
    :param basis: Mapping column -> corresponding basis row.
    """

    # Number of input variables.
    # Equals to length of the row, -1 for the constant, minus the number of
    # basic variables.
    no_vars = len(pivoted_tableau[0]) - 1 - len(basis)

    constants = pivoted_tableau[:, -1]
    def get_variable_value(idx):
        if idx not in basis:

            # Nonbasic variable.
            return 0
        else:
            return constants[basis[idx]]

    return map(get_variable_value, xrange(no_vars))

def get_obj_value(tableau):
    return - tableau[-1][-1]

def two_phase_simplex(initial_constraints, objective_func):
    logging.debug("Constraints: \n %s", initial_constraints)

    # Add auxilary variables.
    basis_size = len(initial_constraints)

    # One row reserved for the constant.
    no_nonbasic_vars = len(initial_constraints[0]) - 1

    tableau = np.insert(
        initial_constraints,
        -1,
        values=np.identity(basis_size),
        axis=1,
    )

    objective_row = np.concatenate(
        (objective_func, np.zeros(basis_size), np.zeros(1)), axis=0
    )

    tableau = np.concatenate(
        (tableau, [objective_row]),
        axis=0
    )

    logging.debug("Generated initial tableau:\n %s", tableau)

    # Now, we only need two-phase simplex if a column of constants has negative
    # values.
    constants = tableau[:, -1]
    has_negatives = len(constants[constants < 0]) > 0

    if not has_negatives:
        basis = {i + no_nonbasic_vars: i for i in xrange(basis_size)}
        logging.debug("Zero solution is feasible, running simplex")
        return simplex_pivoting(tableau, basis, False)

    ## Oh well. Now we have to construct an auxiliary tableau.

    # Add a row of -1's.
    aux_tableau = np.insert(
        tableau,
        0,
        values=-1,
        axis=1
    )

    # Except that the first one in the objective function is zero.
    aux_tableau[-1][0] = 0

    # Add an objective function solving for x_0.
    aux_tableau = np.concatenate(
        (
            aux_tableau,
            [
                np.concatenate(
                    ([-1], np.zeros(len(objective_func) + basis_size + 1)),
                )
            ]
        ),
        axis=0
    )

    logging.debug("Generated auxiliary tableau:\n %s", aux_tableau)

    # The first pivot is special, as we have to get into the feasible domain
    # first before we can run the simplex algorithm.
    first_pivot_row = constants.argmin()  # Index of the most -ve constant.

    # column -> row
    aux_basis = {
        # +1 for the extra row for x_0.
        i + no_nonbasic_vars + 1: i for i in xrange(basis_size)
    }
    logging.debug("initial basis is: %s", aux_basis)

    pivot(aux_tableau, first_pivot_row, 0)

    prev_basis_column = reverse_dict(aux_basis)[first_pivot_row]
    aux_basis[0] = first_pivot_row
    aux_basis.pop(prev_basis_column)

    logging.debug("Auxiliary tableau after the first pivot:\n %s", aux_tableau)

    # Perform simplex on the auxiliary problem.
    aux_tableau, aux_basis, status = simplex_pivoting(
        aux_tableau, aux_basis, extra_obj_func=True
    )

    # The problem is feasible if and only if auxiliary problem is optimal
    # on zero.
    if not status == LP_STATE.OPTIMAL or abs(get_obj_value(aux_tableau)) > EPSILON:
        logging.debug("After solving the auxiliary problem, status = %s", status)
        logging.debug("Objective value = %s", get_obj_value(aux_tableau))
        logging.debug("Auxiliary tableau is\n%s", aux_tableau)
        return (tableau, aux_basis, LP_STATE.UNFEASIBLE)
    logging.debug("Succesfully found the initial state, initializing LP")

    # Convert the auxiliary basis to basis.
    basis = {
        column - 1: row for column, row in aux_basis.iteritems()
    }

    # Cut the tableau out of auxiliary tableau.
    tableau = aux_tableau[:-1:, 1::]
    logging.debug("Extracted tableau: \n%s", tableau)
    return simplex_pivoting(
        tableau, basis, extra_obj_func=False
    )

def simplex_pivoting(tableau, basis, extra_obj_func=False):
    """
    :param tableau: Tableau of the form
        [
            [..., <constant>]
            [objective func, <constant>]
        ]
        NOTE: the first implied column (0, 0, ..., -1) is NOT included.
    :param basis: Mapping column -> row containing "1".
        Corresponds to columns in the simplex basis.
    """
    objective = tableau[-1]
    constraints = tableau[:-1]

    while True:

        # Find the first positive coefficient.
        pivot_column = None
        logging.debug("Objective = %s", objective)
        for column_no, element in enumerate(objective):
            if element > 0:
                pivot_column = column_no
                break

        if pivot_column is None:
            # No positive coefficients: we have optimized the objective
            # function.
            break

        ## For the pivot column, find the smallest fraction.
        constants = constraints[:, -1]

        var_coeffs = constraints[:, pivot_column]

        # Yes, it is "less", not "more", even we do get values which are
        # "more" then zero.
        pos_ve_var_coeffs = np.ma.masked_array(
            data=var_coeffs,
            mask=var_coeffs<0,
            fill_value=np.inf
        )

        if extra_obj_func:
            # Mask second-last row: it's there only to contain a second
            # objective function.
            pos_ve_var_coeffs.mask[-1] = True

        if pos_ve_var_coeffs.count() == 0:

            # No positive coefficients => unbounded.
            return (tableau, basis, LP_STATE.UNBOUNDED)

        logging.debug("Raw var coeffs = %s", var_coeffs)
        logging.debug("Fractions = %s", constants / pos_ve_var_coeffs)
        pivot_row = (constants / pos_ve_var_coeffs).argmin()

        pivot(tableau, pivot_row, pivot_column)

        ## Update the basis status.

        # Pivot column enters the basis.
        old_column = reverse_dict(basis)[pivot_row]
        basis[pivot_column] = pivot_row
        basis.pop(old_column)

    return (tableau, basis, LP_STATE.OPTIMAL)

def pivot(tableau, pivot_row, pivot_column):
    """
    Pivot the tableau on the given row and the given column.
    """
    logging.debug("Pivoting on: row=%s, column=%s", pivot_row, pivot_column)
    # Make this entry equal to <one>.
    mult_coeff = 1 / tableau[pivot_row][pivot_column]
    tableau[pivot_row] *= mult_coeff

    # Make all other entries zero.
    for row_idx in set(xrange(len(tableau))) - {pivot_row}:
        multiplier = tableau[row_idx][pivot_column]
        tableau[row_idx] -= tableau[pivot_row] * multiplier

    logging.debug("Tableau after pivot:\n%s", tableau)

def optimize(lp_in):
    constraints, objective, used_vars = LP_from_stdin(
        StringIO(lp_in), do_prompt=False
    )
    tableau, basis, status = two_phase_simplex(constraints, objective)

    if status == LP_STATE.OPTIMAL:
        values = get_variable_assignments(tableau, basis)
        values_dict = {
            v: values[i] for i, v in enumerate(used_vars)
        }
    else:
        values_dict = {}

    return status, get_obj_value(tableau), values_dict

def main():
    constraints, objective, used_vars = LP_from_stdin(stdin)

    tableau, basis, status = two_phase_simplex(constraints, objective)
    if status == LP_STATE.OPTIMAL:
        print "Solution found!"
        print "Value of the objective function: ", get_obj_value(tableau)

        values = get_variable_assignments(tableau, basis)

        print "Solution: "
        values_dict = {
            v: values[i] for i, v in enumerate(used_vars)
        }
        pprint(values_dict)
    elif status == LP_STATE.UNBOUNDED:
        print "LP Problem is UNBOUNDED"
    elif status == LP_STATE.UNFEASIBLE:
        print "LP Problem is UNFEASIBLE"
    else:
        assert False, "Inconsistent LP state"

def LP_from_stdin(stdin, do_prompt=True):
    """
    :returns: (constraints_matrix, objective_func)
    """
    if do_prompt and stdin.istty():
        prompt = stdout.write
    else:
        # Do not write prompts for non-interactive sessions.
        prompt = lambda text: None

    prompt("Enter objective function (e.g. x + y - 3z): ")
    objective = sympy.poly(
        parse_expr_and_transform(stdin.readline())
    )
    prompt("Enter constraints, one per line, terminate with empty line\n")
    prompt(
        "Constraints are of the form 'x + y - 3z <= 100', only "
        "less than constraints are allowed"
    )
    prompt("NOTE: all variables are assumed to be non-negative\n")
    constraints = []
    while True:
        constraint_raw = stdin.readline()
        if not constraint_raw:
            break
        constraint = parse_expr_and_transform(constraint_raw)

        assert isinstance(
            constraint, LessThan), "Only LessThan constraints are allowed"

        constraint_poly = sympy.poly(
            (constraint.args[0] - constraint.args[1]).simplify()
        )

        assert constraint_poly.is_linear, \
                "Only linear constraints are allowed"

        constraints.append(constraint_poly)

    # Convert constraints and objective func to the matrix form.
    used_vars = sorted(
        set(sum([c.gens for c in constraints], ()) + objective.gens)
    )

    # NOTE: constants do not affect the value of the objective function.
    objective_row = [
        get_coeff_from_poly(objective, var) for var in used_vars
    ]

    constraints_matrix = [
        [
            get_coeff_from_poly(c, var) for var in used_vars
        ] + [-get_constant_from_poly(c)] for c in constraints
    ]

    return (np.array(constraints_matrix), np.array(objective_row), map(str, used_vars))

def get_coeff_from_poly(poly, var):
    """
    Gets coefficient for the variable `var` from the SymPy polynomial `poly`.
    """
    if var not in poly.gens:
        return 0
    return float(poly.coeff_monomial(var))

def get_constant_from_poly(poly):
    """
    Gets the constant value from the given polynomial.
    E. g. "3" for "3 + x - y"
    """
    return float(
        poly.coeff_monomial(tuple(0 for i in xrange(len(poly.gens))))
    )


def parse_expr_and_transform(value):
    """
    Parse_expr curried with basic transformations.
    """
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.sympy_parser import standard_transformations
    from sympy.parsing.sympy_parser import implicit_multiplication_application
    return parse_expr(value, transformations=standard_transformations + (
        implicit_multiplication_application,)
    )

def reverse_dict(d):
    return {
        v: k for k, v in d.iteritems()
    }

if __name__ == "__main__":
    main()
