"""
==============================================================================

Title:             Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models
Subtitle:          Recursive QAOA utilities for weighted Ising models with external fields
Repository:        https://github.com/vijeycreative/NOpt_QAOA1_Tuning
Version:           1.0.0
Date:              02/01/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com
Description
-----------
This module provides:
  • Closed-form p=1 QAOA expectation evaluators for weighted Ising models
    with both pairwise couplings and external fields.
  • Trigonometric coefficient routines that reduce the two-parameter
    p=1 QAOA optimization problem to a one-dimensional search over gamma,
    followed by analytic or semi-analytic recovery of stationary beta values.
  • Utilities for finding real quartic roots and converting them into
    candidate stationary points for beta optimization.
  • Frequency-estimation and line-search helpers used to determine a
    suitable gamma search scale with fewer samples.
  • Recursive QAOA (RQAOA) elimination routines for graphs with external
    fields, including edge-based correlation elimination and direct node
    elimination from strong local biases.

Implementation Notes
--------------------
• Angle convention:
  This implementation uses trigonometric factors of the form cos(2*gamma*w)
  and sin(2*gamma*w). This convention is valid provided it is used
  consistently with the circuit parameterization and Hamiltonian scaling.

• External fields:
  Diagonal entries of the adjacency matrix are interpreted as local field
  terms. These contribute both to the QAOA expectation formulas and to the
  variable elimination rules.

• Optimization strategy:
  The p=1 QAOA cost is reduced to a one-dimensional optimization over gamma.
  For each gamma, stationary beta candidates are obtained from quartic roots
  derived from the trigonometric form of the cost function.

• Performance:
  The core closed-form evaluators and coefficient routines are Numba
  JIT-compiled with `@jit(nopython=True)`.

How to Cite
-----------
If you use this code in academic work, please cite the associated paper:
  Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models
  https://arxiv.org/abs/2501.16419

License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""

import sys
import numpy as np
import networkx as nx
from scipy import optimize
from functools import partial
from numpy import sin, cos
from numba import jit

from fqs import single_quartic
from utils import *
import warnings

# Suppress the specific warning
warnings.filterwarnings('ignore', message="delta_grad == 0.0. Check if the approximated function is linear.")


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

@jit(nopython=True)
def QAOA_Expectation_Fields_Cost(edges, adj_mat, angles):
    """
    Compute the total p=1 QAOA objective for a weighted Ising model with external fields.

    This routine evaluates the closed-form p=1 QAOA expectation value for an
    Ising Hamiltonian containing both pairwise interaction terms and local field
    terms. Edge contributions are computed from correlator-like quantities for
    each interaction term, while node contributions are computed from the local
    field expectation values.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list of the reduced graph. Each entry corresponds to a pairwise
        interaction term in the Ising Hamiltonian.
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric matrix containing the Ising couplings. Off-diagonal entries
        store pairwise couplings, and diagonal entries store local field terms.
    angles : array-like of length 2
        QAOA angles ``(gamma, beta)`` for the p=1 circuit.

    Returns
    -------
    float
        Total QAOA expectation value under this module's sign and scaling
        convention.

    Notes
    -----
    • This routine includes both edge and node contributions.
    • Diagonal entries ``adj_mat[i, i]`` are interpreted as external fields.
    • The trigonometric convention must remain consistent with the rest of the
      codebase and the corresponding circuit definition.
    """
    gamma, beta = angles
    edge_costs = {}

    for u, v in edges:

        EX = np.nonzero(adj_mat[v])[0]
        eX = EX[np.where(EX != v)]
        e = eX[np.where(eX != u)]

        DX = np.nonzero(adj_mat[u])[0]
        dX = DX[np.where(DX != u)]
        d = dX[np.where(dX != v)]

        F = np.intersect1d(e, d)

        term1_cos1 = cos(2 * gamma * adj_mat[v, v])
        for x in e:
            term1_cos1 *= cos(2 * gamma * adj_mat[x, v])
        term1_cos2 = cos(2 * gamma * adj_mat[u, u])
        for y in d:
            term1_cos2 *= cos(2 * gamma * adj_mat[u, y])
        term1 = sin(4 * beta) * sin(2 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)

        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))

        # Compute the second term in the square brackets of Eq 13.
        term2 = -np.power(sin(2 * beta), 2)
        for x, y in E:
            term2 *= cos(2 * gamma * adj_mat[x, y])
        triangle_1_terms = cos(2 * gamma * (adj_mat[u, u] + adj_mat[v, v]))
        triangle_2_terms = cos(2 * gamma * (adj_mat[u, u] - adj_mat[v, v]))
        for f in F:
            triangle_1_terms *= cos(2 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2 * gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 = term2 * (triangle_1_terms - triangle_2_terms)

        ZuZv = term1 + term2
        edge_costs[(u, v)] = (adj_mat[u, v] / 2) * ZuZv

    node_costs = {}

    for i in range(adj_mat.shape[0]):

        if adj_mat[i, i] != 0:
            c_i = sin(2 * beta) * sin(2 * gamma * adj_mat[i, i])

            EX = np.nonzero(adj_mat[i])[0]
            eX = EX[np.where(EX != i)]

            for x in eX:
                c_i *= cos(2 * gamma * adj_mat[x, i])

            node_costs[i] = adj_mat[i, i] * c_i
        else:
            node_costs[i] = 0.0

    total_cost = 0
    for v in edge_costs.values():
        total_cost += v
    for v in node_costs.values():
        total_cost += v
    return total_cost


@jit(nopython=True)
def QAOA_Expectation_Fields_Edges(edges, adj_mat, angles):
    """
    Compute per-edge and per-node expectation scores for p=1 QAOA with external fields.

    This routine is similar to `QAOA_Expectation_Fields_Cost`, but instead of
    returning the total objective value, it returns the individual edge and node
    expectation-like quantities used during variable elimination in RQAOA.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list of the reduced graph.
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric matrix containing pairwise couplings and local field terms.
        Off-diagonal entries represent edge weights, and diagonal entries
        represent node weights.
    angles : array-like of length 2
        QAOA angles ``(gamma, beta)`` for the p=1 circuit.

    Returns
    -------
    (dict, dict)
        A pair ``(edge_costs, node_costs)`` where:
          • ``edge_costs[(u, v)]`` stores the correlator-like score associated
            with edge ``(u, v)``.
          • ``node_costs[i]`` stores the local-field expectation-like score
            associated with node ``i``.

    Notes
    -----
    These scores are typically used to determine whether the next RQAOA
    elimination step should remove an edge by correlation rounding or remove
    a node directly due to a strong local bias.
    """
    gamma, beta = angles
    edge_costs = {}

    for u, v in edges:

        EX = np.nonzero(adj_mat[v])[0]
        eX = EX[np.where(EX != v)]
        e = eX[np.where(eX != u)]

        DX = np.nonzero(adj_mat[u])[0]
        dX = DX[np.where(DX != u)]
        d = dX[np.where(dX != v)]

        F = np.intersect1d(e, d)

        term1_cos1 = cos(2 * gamma * adj_mat[v, v])
        for x in e:
            term1_cos1 *= cos(2 * gamma * adj_mat[x, v])
        term1_cos2 = cos(2 * gamma * adj_mat[u, u])
        for y in d:
            term1_cos2 *= cos(2 * gamma * adj_mat[u, y])
        term1 = sin(4 * beta) * sin(2 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)

        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))

        # Compute the second term in the square brackets of Eq 13.
        term2 = -np.power(sin(2 * beta), 2)
        for x, y in E:
            term2 *= cos(2 * gamma * adj_mat[x, y])
        triangle_1_terms = cos(2 * gamma * (adj_mat[u, u] + adj_mat[v, v]))
        triangle_2_terms = cos(2 * gamma * (adj_mat[u, u] - adj_mat[v, v]))
        for f in F:
            triangle_1_terms *= cos(2 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2 * gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 = term2 * (triangle_1_terms - triangle_2_terms)

        ZuZv = term1 + term2
        edge_costs[(u, v)] = 0.5 * ZuZv

    node_costs = {}

    for i in range(adj_mat.shape[0]):

        if adj_mat[i, i] != 0:
            c_i = sin(2 * beta) * sin(2 * gamma * adj_mat[i, i])

            EX = np.nonzero(adj_mat[i])[0]
            eX = EX[np.where(EX != i)]

            for x in eX:
                c_i *= cos(2 * gamma * adj_mat[x, i])

            node_costs[i] = c_i
        else:
            node_costs[i] = 0.0

    return edge_costs, node_costs


# ---------------------------------------------------------------------------
# Trigonometric reduction helpers
# ---------------------------------------------------------------------------

@jit(nopython=True)
def QAOA_Cost_Coefficients(edges, adj_mat, gamma):
    """
    Compute the total trigonometric coefficients used in the reduced p=1 objective.

    For a fixed gamma, the p=1 QAOA cost can be written in a reduced
    trigonometric form involving coefficients ``A``, ``B``, and ``C``. These
    coefficients are then used to determine stationary points in beta via a
    quartic equation.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list of the current graph instance.
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric matrix containing pairwise couplings in the off-diagonal
        entries and local field terms in the diagonal entries.
    gamma : float
        QAOA gamma angle.

    Returns
    -------
    (float, float, float)
        Tuple ``(term_A, term_B, term_C)`` giving the total coefficients of the
        reduced trigonometric cost function.
    """
    term_A = 0
    term_B = 0
    term_C = 0
    for u, v in edges:
        # TODO: maybe check if J_uv = 0? If so then skip
        EX = np.nonzero(adj_mat[v])[0]
        eX = EX[np.where(EX != v)]
        e = eX[np.where(eX != u)]  # edges connected to u (excl v)

        DX = np.nonzero(adj_mat[u])[0]
        dX = DX[np.where(DX != u)]
        d = dX[np.where(dX != v)]  # edges connected to v (excl u)

        F = np.intersect1d(e, d)  # edges connected to both u & v

        ####################
        # 1a. b_ij (in front of sin(4beta))
        term1_cos1 = cos(2 * gamma * adj_mat[v, v])
        for x in e:  # product in place
            term1_cos1 *= cos(2 * gamma * adj_mat[x, v])

        term1_cos2 = cos(2 * gamma * adj_mat[u, u])
        for y in d:
            term1_cos2 *= cos(2 * gamma * adj_mat[u, y])

        term1 = (adj_mat[u, v] / 2) * sin(2 * gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)
        term_B += term1  # add into combined coefficient
        ####################
        # 1b. c_ij
        # all edges connected to x OR y that are NOT connected to each other (does not form triangle)
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))

        # Compute the second term in the square brackets
        term2 = -1  # the negative sign's here
        for x, y in E:
            term2 = term2 * cos(2 * gamma * adj_mat[x, y])
        triangle_1_terms = cos(2 * gamma * (adj_mat[u, u] + adj_mat[v, v]))
        triangle_2_terms = cos(2 * gamma * (adj_mat[u, u] - adj_mat[v, v]))
        for f in F:
            triangle_1_terms *= cos(2 * gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(2 * gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 *= (adj_mat[u, v] / 2) * (triangle_1_terms - triangle_2_terms)
        term_C += term2
        ####################

    # 2. Linear terms a_i(gamma)
    for i in range(adj_mat.shape[0]):
        if adj_mat[i, i] != 0:
            c_i = sin(2 * gamma * adj_mat[i, i])
            EX = np.nonzero(adj_mat[i])[0]
            eX = EX[np.where(EX != i)]  # edges connected to i
            for x in eX:
                c_i *= cos(2 * gamma * adj_mat[i, x])

            # multiply by leading coefficient and add to total
            term_A += adj_mat[i, i] * c_i
        else:
            continue  # don't add anything if linear term was 0
    return term_A, term_B, term_C


def find_roots(A, B, C):
    """
    Find the real roots of the quartic polynomial induced by the beta stationary condition.

    The reduced p=1 optimization over beta leads to a quartic polynomial whose
    real roots determine candidate stationary points. This helper computes those
    roots, preferring the custom `single_quartic` routine and falling back to
    `numpy.roots` if needed.

    Parameters
    ----------
    A : float
        First trigonometric coefficient.
    B : float
        Second trigonometric coefficient.
    C : float
        Third trigonometric coefficient.

    Returns
    -------
    list[float]
        List of real roots, filtered by discarding roots whose imaginary parts
        are not numerically negligible.
    """

    def sq(el):
        return np.power(el, 2)

    # Define the polynomial coefficients in descending order of power
    poly = np.array([
        16 * sq(B) + 4 * sq(C),              # Coefficient of x^4
        8 * A * B,                           # Coefficient of x^3
        sq(A) - 16 * sq(B) - 4 * sq(C),     # Coefficient of x^2
        -4 * A * B,                          # Coefficient of x^1
        4 * sq(B)                            # Coefficient of x^0
    ])

    # Compute the roots of the polynomial
    roots = single_quartic(*poly)

    # Filter out complex roots by checking if the imaginary part is close to zero
    filtered_roots = [np.real(r) for r in roots if np.isclose(np.imag(r), 0, atol=1e-3)]
    if not filtered_roots:
        roots = np.roots(poly)
        filtered_roots = [np.real(r) for r in roots if np.isclose(np.imag(r), 0, atol=1e-3)]

    return filtered_roots


@jit(nopython=True)
def beta_stat_points(roots):
    """
    Convert real quartic roots into candidate stationary points for beta.

    Each real root is interpreted as an argument of an inverse cosine,
    generating up to two beta stationary points. These points are returned as
    candidate angles to test in the reduced p=1 QAOA objective.

    Parameters
    ----------
    roots : array-like
        One-dimensional array or list of real quartic roots.

    Returns
    -------
    numpy.ndarray
        Array of candidate stationary beta values.
    """
    # Initialize an array to hold the stationary points (up to 8 points)
    stat_points = np.zeros(len(roots) * 2)

    for i, r in enumerate(roots):
        # Check if the root is outside the valid domain [-1, 1]
        if r > 1.0 or r < -1.0:
            # If it's outside the bounds, set the stationary point to 0
            t = 0
        else:
            # Calculate the angle t using arccos for valid roots
            t = np.real(np.arccos(r))

        # Calculate the two stationary points for each root
        stat_points[2 * i] = -0.5 * t
        stat_points[2 * i + 1] = 0.5 * t

    return stat_points


def eval_qaoa_gamma(edges, adj_mat, gamma) -> tuple:
    """
    Evaluate the reduced p=1 QAOA objective at a fixed gamma.

    For a given gamma, this function computes the reduced trigonometric
    coefficients, determines the candidate stationary points in beta, evaluates
    the reduced beta objective at those points, and returns the best cost along
    with the corresponding beta.

    Parameters
    ----------
    edges : numpy.ndarray
        Edge list of the current graph instance.
    adj_mat : numpy.ndarray
        Symmetric matrix containing the Ising couplings and field terms.
    gamma : float or numpy.ndarray
        Candidate gamma value. If an array is passed, the first element is used.

    Returns
    -------
    (float, float)
        Tuple ``(opt_cost_along_beta, opt_beta)`` giving the best reduced cost
        value found along beta and its corresponding beta angle.
    """

    # Handle case where gamma is an ndarray
    if isinstance(gamma, np.ndarray):
        gamma = gamma[0]

    # Get QAOA cost coefficients
    qaoa_coeffs = QAOA_Cost_Coefficients(edges, adj_mat, gamma)

    try:
        # Find roots of the polynomial
        quart_roots = find_roots(*qaoa_coeffs)
    except ZeroDivisionError:
        # Handle division by zero error
        return 0.0, 0.0

    if not quart_roots:
        opt_cost_along_beta, opt_beta = 0.0, 0.0
    else:
        # Compute beta stationary points
        beta_points = beta_stat_points(quart_roots)

        # Define the polynomial H(beta) explicitly as a lambda function
        def beta_poly(A, B, C, beta):
            return A * np.sin(2 * beta) + B * np.sin(4 * beta) + C * np.sin(2 * beta) ** 2

        # Evaluate the polynomial at the beta stationary points
        beta_values = beta_poly(*qaoa_coeffs, beta_points)

        # Find the index of the minimum value in beta_values
        opt_beta_index = np.argmin(beta_values)

        # Get the optimal cost and beta value
        opt_cost_along_beta = beta_values[opt_beta_index]
        opt_beta = beta_points[opt_beta_index]

    return opt_cost_along_beta, opt_beta


def eval_qaoa_gamma_bare(edges, adj_mat, gamma) -> float:
    """
    Wrapper around `eval_qaoa_gamma` returning only the gamma-dependent cost.

    This helper is used during one-dimensional gamma optimization. It also
    enforces the correct boundary behavior by returning zero at angles that are
    numerically close to the domain boundaries.

    Parameters
    ----------
    edges : numpy.ndarray
        Edge list of the current graph instance.
    adj_mat : numpy.ndarray
        Symmetric matrix containing the Ising couplings and field terms.
    gamma : float
        Candidate gamma value.

    Returns
    -------
    float
        Reduced objective value evaluated at the given gamma.
    """
    if np.isclose(gamma, 0.0) or np.isclose(gamma, np.pi / 2) or np.isclose(gamma, np.pi):
        return 0.0
    else:
        return eval_qaoa_gamma(edges, adj_mat, gamma)[0]


@jit(nopython=True)
def get_max_frequency_fields(edges, adj_mat):
    """
    Estimate a dominant trigonometric frequency scale for the p=1 QAOA objective.

    This heuristic is used to choose a reasonable search spacing or initial
    guess when optimizing gamma. The estimate accounts for both pairwise
    interaction terms and local field contributions.

    Parameters
    ----------
    edges : numpy.ndarray, shape (m, 2)
        Edge list of the graph instance.
    adj_mat : numpy.ndarray, shape (n, n)
        Symmetric matrix containing pairwise couplings in the off-diagonal
        entries and local field terms in the diagonal entries.

    Returns
    -------
    float
        Maximum estimated frequency scale across the current graph instance.
    """
    edge_frequencies = {}

    for u, v in edges:
        # Find neighbors of v excluding v itself
        EX = np.nonzero(adj_mat[v])[0]
        eX = EX[EX != v]
        e = eX[eX != u]

        # Calculate frequency contribution from v
        e_freq = adj_mat[v, v] + np.sum(np.array([np.abs(adj_mat[v, w]) for w in e]))

        # Find neighbors of u excluding u itself
        DX = np.nonzero(adj_mat[u])[0]
        dX = DX[DX != u]
        d = dX[dX != v]

        # Calculate frequency contribution from u
        d_freq = adj_mat[u, u] + np.sum(np.array([np.abs(adj_mat[u, w]) for w in d]))

        max_term1_freq = 2 * (adj_mat[u, v] + max(e_freq, d_freq))

        # Find common neighbors between e and d (forming a triangle)
        F = np.intersect1d(e, d)

        # Non-triangle edges
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = list(set(e_edges_non_triangle + d_edges_non_triangle))

        # Calculate leading frequency terms
        leading_freq = np.sum(np.array([np.abs(adj_mat[x, y]) for x, y in E]))
        triangle_1_terms = np.abs(adj_mat[u, u] + adj_mat[v, v])
        triangle_2_terms = np.abs(adj_mat[u, u] - adj_mat[v, v])
        for f in F:
            triangle_1_terms += np.abs(adj_mat[u, f] + adj_mat[v, f])
            triangle_2_terms += np.abs(adj_mat[u, f] - adj_mat[v, f])

        max_term2_freq = 2 * (leading_freq + max(triangle_1_terms, triangle_2_terms))

        edge_frequencies[(u, v)] = max(max_term1_freq, max_term2_freq)

    node_requencies = {}

    for i in range(adj_mat.shape[0]):

        if adj_mat[i, i] != 0:
            c_i = adj_mat[i, i]

            EX = np.nonzero(adj_mat[i])[0]
            eX = EX[np.where(EX != i)]

            for x in eX:
                c_i += np.abs(adj_mat[x, i])

            node_requencies[i] = 2 * c_i
        else:
            node_requencies[i] = 0.0

    max_edge_frequency = max(edge_frequencies.values())
    max_node_frequency = max(node_requencies.values())

    return max(max_edge_frequency, max_node_frequency)


def lazy_line_search(func, spacing):
    """
    Perform a simple forward line search to bracket a local minimum.

    Starting from a given spacing, this routine evaluates the objective at
    equally spaced points until the function stops decreasing. The resulting
    interval is then used as a search bracket for a more refined scalar
    optimizer.

    Parameters
    ----------
    func : callable
        One-dimensional function to minimize.
    spacing : float
        Step size used in the initial forward scan.

    Returns
    -------
    (float, float)
        Interval ``(point_a, point_b)`` that brackets the local minimum found
        by the scan.
    """
    min_cost = func(spacing)
    step = 1

    while True:
        step += 1
        cost = func(spacing * step)

        if cost < min_cost:
            min_cost = cost
        else:
            break

    # Define the search interval for bisection optimization
    point_a = spacing * (step - 2)
    point_b = spacing * step

    return point_a, point_b


def Optimise_QAOA(red_edges: np.array, adj_mat: np.ndarray, n_samps) -> tuple:
    """
    Optimize the p=1 QAOA angles for a graph with external fields.

    This routine reduces the two-parameter optimization problem to a
    one-dimensional search over gamma. If `n_samps` is provided, it performs a
    brute-force grid search over gamma. Otherwise, it estimates a characteristic
    frequency scale, performs a coarse line search to bracket a minimum, and
    then refines gamma with a bounded scalar optimizer. The corresponding beta
    is then recovered from the stationary-point analysis.

    Parameters
    ----------
    red_edges : numpy.ndarray
        Edge list of the reduced graph instance.
    adj_mat : numpy.ndarray
        Symmetric matrix containing pairwise couplings and local field terms.
    n_samps : int or None
        Number of gamma samples for brute-force search. If `None`, use the
        adaptive line-search / bounded-optimization strategy.

    Returns
    -------
    (list[float], float)
        Tuple ``(optimal_angles, qaoa_cost)`` where:
          • ``optimal_angles = [opt_gamma, opt_beta]``
          • ``qaoa_cost`` is the reduced p=1 QAOA objective value at those angles.
    """

    # Create a partial function with fixed red_edges and adj_mat
    qaoa_partial = partial(eval_qaoa_gamma_bare, red_edges, adj_mat)

    # Determine the maximal frequency in the cost function
    max_freq = get_max_frequency_fields(red_edges, adj_mat)

    if n_samps is not None:
        opt_gamma = optimize.brute(qaoa_partial, ((0, np.pi),), Ns=n_samps, workers=1)
        opt_gamma = opt_gamma[0]
    else:
        initial_guess = 1 / (2 * max_freq + 1)
        point_a, point_b = lazy_line_search(qaoa_partial, initial_guess)
        result = optimize.minimize_scalar(
            qaoa_partial,
            bounds=(point_a, point_b),
            method='bounded',
            options={'maxiter': 900000, 'xatol': 0.25 * initial_guess, 'disp': 1}
        )
        opt_gamma = result.x

    # Get the QAOA cost for the optimized gamma
    qaoa_cost = qaoa_partial(opt_gamma)

    # Determine the optimal beta for the optimized gamma
    _, optimal_beta = eval_qaoa_gamma(red_edges, adj_mat, opt_gamma)
    optimal_angles = [opt_gamma, optimal_beta]

    return optimal_angles, qaoa_cost


# ---------------------------------------------------------------------------
# RQAOA elimination + driver
# ---------------------------------------------------------------------------

def eliminate_variable(graphmanager: GraphManager, n_samps, onlynode):
    """
    Perform one RQAOA elimination step for an Ising model with external fields.

    This routine optimizes the p=1 QAOA angles on the current reduced graph,
    evaluates both edge-based and node-based expectation scores, and then
    performs one elimination step. Depending on the relative magnitudes of the
    node and edge scores, the step may either:
      • eliminate a node directly using its local bias, or
      • correlate / anti-correlate an edge and eliminate one of its endpoints.

    Parameters
    ----------
    graphmanager : GraphManager
        Stateful manager containing the reduced graph, logs, and elimination
        methods.
    n_samps : int or None
        Number of samples for brute-force gamma optimization. If `None`, the
        adaptive optimization route is used.
    onlynode : bool
        If `True`, force the elimination step to consider only direct node
        elimination based on local biases.

    Returns
    -------
    None
        The graph manager is mutated in-place.
    """
    red_edges, adj_mat = extract_properties(graphmanager)

    # RUN p = 1QAOA
    optimal_angles, qaoa_cost = Optimise_QAOA(red_edges, adj_mat, n_samps)

    # Now do correlation rounding management
    graphmanager.optimal_angles[graphmanager.iter] = optimal_angles
    # Get the beta value corresponding to this value of gamma
    # Get the correlation b/w each edge and the sort the edges with in descending order of the magnitude of correlation.
    edge_costs, node_costs = QAOA_Expectation_Fields_Edges(red_edges, adj_mat, optimal_angles)

    # TODO: do we need this sort? maybe just take max instead
    edge_costs = {k: v for k, v in sorted(edge_costs.items(), key=lambda item: np.abs(item[1]), reverse=True)}
    node_costs = {k: v for k, v in sorted(node_costs.items(), key=lambda item: np.abs(item[1]), reverse=True)}

    # Get the edge with highest correlation
    edge, edge_weight = list(edge_costs.items())[0]
    node, node_weight = list(node_costs.items())[0]

    # if the cost of the node itself is greater than the max corr. in the edges, directly round according to the node bias
    if onlynode or (np.abs(node_weight) >= np.abs(edge_weight)):

        node_sign = int(np.sign(sys.float_info.epsilon + node_weight))
        node_val = 1 if node_sign > 0 else -1
        msg1 = f"QAOA Cost = {qaoa_cost}. Removing Node {node} that has maximum absolute weight {node_weight} by assigning value {node_val}."
        graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + msg1 + "\n"
        if graphmanager.verbose:
            print(msg1)
        graphmanager.eliminate_node(node, node_val)
    else:
        # Get the sign of the correlation; this is either +1 or -1.
        edge_sign = int(np.sign(sys.float_info.epsilon + edge_weight))

        num_triangles_u = nx.triangles(graphmanager.reduced_graph)[edge[0]]
        num_triangles_v = nx.triangles(graphmanager.reduced_graph)[edge[1]]

        # Either correlate or anti-correlate depending on the sign.
        if edge_sign < 0:
            msg1 = f"QAOA Cost = {qaoa_cost}. Anti-Correlating Edge {edge} that has maximum absolute weight {edge_weight}."
            msg2 = f"Node {edge[0]} and {edge[1]} were contained in {num_triangles_u} and {num_triangles_v} triangles respectively."
            graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + msg1 + "\n" + msg2 + "\n"
            if graphmanager.verbose:
                print(msg1)
                print(msg2)
            graphmanager.anti_correlate(edge)
        elif edge_sign > 0:
            msg1 = f"QAOA Cost = {qaoa_cost}. Correlating Edge {edge} that has maximum absolute weight {edge_weight}."
            msg2 = f"Node {edge[0]} and {edge[1]} were contained in {num_triangles_u} and {num_triangles_v} triangles respectively."
            graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + msg1 + "\n" + msg2 + "\n"
            if graphmanager.verbose:
                print(msg1)
                print(msg2)
            graphmanager.correlate(edge)
        else:
            error_msg = f"Cannot correlate or anti-correlate edge {edge} for weight {edge_weight}."
            graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + error_msg + "\n"
            if graphmanager.verbose:
                print(error_msg)


def RQAOA_Fields(graphmanager: GraphManager, n, n_samps=None, only_node=False, draw=False):
    """
    Run RQAOA on an Ising model with external fields for up to `n` elimination steps.

    At each iteration, this driver applies one elimination step to the current
    reduced graph. The procedure stops early if no edges remain. After the
    elimination stage is complete, the residual problem is solved exactly using
    the graph manager's brute-force solver.

    Parameters
    ----------
    graphmanager : GraphManager
        Stateful manager for the original graph, reduced graph, logs, and
        elimination operations.
    n : int
        Maximum number of elimination steps to perform.
    n_samps : int or None, optional
        Number of brute-force samples for gamma optimization. If `None`,
        use the adaptive optimization route.
    only_node : bool, optional
        If `True`, each elimination step will prefer direct node elimination
        based on local field bias.
    draw : bool, optional
        If `True`, draw the reduced graph with node fields after each
        elimination step.

    Returns
    -------
    Any
        Whatever is returned by `graphmanager.brute_force()`, typically the
        best residual cost and reconstructed assignment.
    """
    # Perform Variable Elimation until i == n.
    i = 0
    while i < n:
        if graphmanager.reduced_graph.number_of_edges() == 0:
            break
        out_message = f"Iter {i}: Graph has {graphmanager.reduced_graph.number_of_nodes()} nodes and {graphmanager.reduced_graph.number_of_edges()} edges remaining."
        graphmanager.log[graphmanager.iter] = out_message + "\n"
        if graphmanager.verbose:
            print(out_message)
        eliminate_variable(graphmanager, n_samps, only_node)
        i += 1
        graphmanager.iter += 1
        if draw:
            draw_graph_with_fields(graphmanager.reduced_graph)
    # After eliminating 'n' variables, solve the reduced problem using brute-force
    graphmanager.log[graphmanager.iter] = "\nBrute-Forcing\n"
    if graphmanager.verbose:
        print("\nBrute-Forcing")
    return graphmanager.brute_force()