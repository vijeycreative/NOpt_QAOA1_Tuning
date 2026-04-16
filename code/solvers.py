"""
==============================================================================

Title:             Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models
Subtitle:          Classical exact and SDP-based benchmark solvers for Ising and Max-Cut instances
Repository:        https://github.com/vijeycreative/NOpt_QAOA1_Tuning
Version:           1.0.0
Date:              16/04/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • A semidefinite programming (SDP) relaxation routine for estimating ground
    states of 2-local Ising models, together with random hyperplane rounding
    to extract candidate spin assignments.
  • Exact mixed-integer optimization solvers for weighted Max-Cut instances
    using Gurobi.
  • Exact mixed-integer optimization solvers for 2-local Ising models with
    pairwise couplings only.
  • Exact mixed-integer optimization solvers for 2-local Ising models with
    both pairwise couplings and external field terms.
  • Utility routines for returning both the optimal objective value and the
    corresponding spin assignment in the {-1, +1} convention.

Implementation Notes
--------------------
• Spin convention:
  Internally, Gurobi binary variables are converted to Ising spins through
  the mapping z = 2x - 1, so that binary values {0, 1} correspond to spin
  values {-1, +1}.

• SDP relaxation:
  The SDP-based solver optimizes over a positive semidefinite matrix with
  unit diagonal entries, then applies repeated random hyperplane rounding
  to obtain candidate discrete spin configurations.

• External fields:
  In the Ising-with-fields solver, node attributes named 'weight' are treated
  as local field coefficients h_i in the objective.

• Performance:
  The Gurobi-based solvers return exact or best-found solutions together with
  the corresponding MIP gap, and optionally allow control over verbosity,
  time limits, and thread count.

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

import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import numpy as np
import networkx as nx
from scipy.linalg import sqrtm


def find_ground_state_sdp(adj_matrix, n=1024):
    """
    Estimate low-energy spin configurations of a 2-local Ising model using SDP.

    This routine solves the standard semidefinite relaxation of a 2-local Ising
    problem and then performs repeated random hyperplane rounding to extract
    candidate spin assignments in {-1, +1}. The returned list contains all
    rounded energies and their corresponding spin configurations.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Symmetric weighted adjacency matrix representing the Ising model.
        Off-diagonal entries encode pairwise couplings.
    n : int, optional
        Number of random hyperplane rounding iterations to perform.

    Returns
    -------
    (list[float], list[list[int]])
        A tuple ``(all_costs, all_spins)`` where:
          • ``all_costs`` is the list of rounded energies obtained across all
            hyperplane samples.
          • ``all_spins`` is the list of corresponding spin assignments, with
            each spin taking values in {-1, +1}.
    """
    # Number of nodes
    n_nodes = adj_matrix.shape[0]
    
    # Create a symmetric matrix variable for the SDP
    X = cp.Variable((n_nodes, n_nodes), symmetric=True)
    
    # Constraints: X must be positive semidefinite and the diagonal elements must be 1
    constraints = [X >> 0]  # X is positive semidefinite
    constraints += [X[i, i] == 1 for i in range(n_nodes)]  # Diagonal elements are 1
    
    # Objective function: minimize sum of edge weights times X_ij (reflecting spin-spin interactions)
    objective = cp.Minimize(cp.sum(cp.multiply(adj_matrix, X)))
    
    # Define the SDP problem
    prob = cp.Problem(objective, constraints)
    
    # Solve the SDP problem
    prob.solve()
    
    # Extract the optimal matrix X
    X_opt = X.value
    
    # Initialize lists to store all costs and corresponding spins
    all_costs = []
    all_spins = []
    
    # Perform n rounds of random hyperplane rounding
    for _ in range(n):
        # Take the square root of X_opt
        sqrt_X = sqrtm(X_opt)
        
        # Generate a random hyperplane
        u = np.random.randn(n_nodes)
        
        # Assign spins based on the sign of the projection onto the hyperplane
        spins = np.sign(sqrt_X @ u)
        
        # Convert spins to real numbers (-1 or 1)
        spins = np.where(spins >= 0, 1, -1).astype(int).tolist()
        
        # Calculate the cost for the current spin configuration
        cost = 0.5 * np.sum(adj_matrix * np.outer(spins, spins))
        
        # Store the current cost and corresponding spins
        all_costs.append(cost)
        all_spins.append(spins)
    
    return all_costs, all_spins


def find_maxcut(graph, verbose=True, tlimit=None, max_cores=None):
    """
    Solve a weighted Max-Cut instance exactly using Gurobi.

    The optimization is formulated using binary decision variables which are
    converted to Ising-like spin variables in {-1, +1}. The objective maximizes
    the weighted cut value over all graph edges.

    Parameters
    ----------
    graph : nx.Graph
        Weighted graph for which the Max-Cut problem is to be solved. Edge
        weights are expected in the ``'weight'`` attribute.
    verbose : bool, optional
        If ``False``, suppress Gurobi solver output.
    tlimit : int or float or None, optional
        Time limit for the optimization in seconds. If ``None``, no time limit
        is imposed.
    max_cores : int or None, optional
        Maximum number of solver threads to use. If ``None``, Gurobi chooses
        its default thread count.

    Returns
    -------
    (float, float, list[int])
        A tuple ``(cost, mipgap, solution_list)`` where:
          • ``cost`` is the optimal or best-found Max-Cut value,
          • ``mipgap`` is the corresponding MIP gap,
          • ``solution_list`` is the spin assignment sorted by node index,
            using values in {-1, +1}.
    """
    # Create a new Gurobi model
    model = gp.Model("MaxCut")

    if not verbose:
        # Suppress Gurobi output
        model.Params.OutputFlag = 0
    
    # Add binary variables for each node
    spins = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="spin")
    
    # Convert binary variables to -1 and 1
    spin_expr = {node: 2 * spins[node] - 1 for node in graph.nodes()}
    
    # Set objective: maximize sum_uv J_uv * (1 - Z_u * Z_v) / 2
    objective = gp.quicksum(graph[u][v]['weight'] * (1 - spin_expr[u] * spin_expr[v]) / 2 for u, v in graph.edges())
    model.setObjective(objective, GRB.MAXIMIZE)
    
    # Set time limit if provided
    if tlimit is not None:
        model.setParam('TimeLimit', tlimit)
    
    # Set the maximum number of cores if provided
    if max_cores is not None:
        model.setParam('Threads', max_cores)
    
    # Optimize the model
    model.optimize()
    
    # Extract the optimal cost and MIP gap
    cost = model.ObjVal
    mipgap = model.MIPGap
    
    # Extract the solution and create the solution list with values -1 and 1
    solution = {node: int(spins[node].X) for node in graph.nodes()}
    solution_list = [2 * solution[node] - 1 for node in sorted(graph.nodes())]  # Convert to -1 and 1
    
    return cost, mipgap, solution_list


def find_ground_state_ising(graph, verbose=True, tlimit=None, max_cores=None):
    """
    Solve a 2-local Ising ground-state problem exactly using Gurobi.

    This routine minimizes the Ising energy consisting only of pairwise
    interaction terms over spin assignments in {-1, +1}.

    Parameters
    ----------
    graph : nx.Graph
        Graph representing the Ising model. Edge weights stored in the
        ``'weight'`` attribute are interpreted as pairwise couplings.
    verbose : bool, optional
        If ``False``, suppress Gurobi solver output.
    tlimit : int or float or None, optional
        Time limit for the optimization in seconds. If ``None``, no time limit
        is imposed.
    max_cores : int or None, optional
        Maximum number of solver threads to use. If ``None``, Gurobi chooses
        its default thread count.

    Returns
    -------
    (float, float, list[int])
        A tuple ``(cost, mipgap, solution_list)`` where:
          • ``cost`` is the optimal or best-found Ising energy,
          • ``mipgap`` is the corresponding MIP gap,
          • ``solution_list`` is the spin assignment sorted by node index,
            using values in {-1, +1}.
    """
    # Create a new Gurobi model
    model = gp.Model("2-local Ising model")

    if not verbose:
        # Suppress Gurobi output
        model.Params.OutputFlag = 0
    
    # Add binary variables for each node (spin variables Z_u)
    spins = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="spin")
    
    # Convert binary variables to -1 and 1
    spin_expr = {node: 2 * spins[node] - 1 for node in graph.nodes()}
    
    # Set objective: minimize sum_uv J_uv Z_u Z_v
    objective = gp.quicksum(graph[u][v]['weight'] * spin_expr[u] * spin_expr[v] for u, v in graph.edges())
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Set time limit if provided
    if tlimit is not None:
        model.setParam('TimeLimit', tlimit)
    
    # Set the maximum number of cores if provided
    if max_cores is not None:
        model.setParam('Threads', max_cores)
    
    # Optimize the model
    model.optimize()
    
    # Extract the optimal cost and MIP gap
    cost = model.ObjVal
    mipgap = model.MIPGap
    
    # Extract the solution and create the solution list
    solution = {node: int(spins[node].X) for node in graph.nodes()}
    solution_list = [2 * solution[node] - 1 for node in sorted(graph.nodes())]  # Convert to -1 and 1
    
    return cost, mipgap, solution_list


def find_ground_state_ising_with_fields(graph, verbose=True, tlimit=None, max_cores=None):
    """
    Solve a 2-local Ising model with external fields exactly using Gurobi.

    This routine minimizes the full Ising energy consisting of both pairwise
    interaction terms and linear field terms. Edge weights are interpreted as
    couplings J_ij, while node weights are interpreted as local fields h_i.

    Parameters
    ----------
    graph : nx.Graph
        Graph representing the Ising model. Edge weights in the ``'weight'``
        attribute are interpreted as pairwise couplings, and node weights in
        the ``'weight'`` attribute are interpreted as local fields.
    verbose : bool, optional
        If ``False``, suppress Gurobi solver output.
    tlimit : int or float or None, optional
        Time limit for the optimization in seconds. If ``None``, no time limit
        is imposed.
    max_cores : int or None, optional
        Maximum number of solver threads to use. If ``None``, Gurobi chooses
        its default thread count.

    Returns
    -------
    (float, float, list[int])
        A tuple ``(cost, mipgap, solution_list)`` where:
          • ``cost`` is the optimal or best-found Ising energy,
          • ``mipgap`` is the corresponding MIP gap,
          • ``solution_list`` is the spin assignment sorted by node index,
            using values in {-1, +1}.
    """
    # Create a new Gurobi model
    model = gp.Model("2-local Ising model with fields")

    if not verbose:
        # Suppress Gurobi output
        model.Params.OutputFlag = 0
    
    # Add binary variables for each node (spin variables Z_u)
    spins = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="spin")
    
    # Convert binary variables to -1 and 1
    spin_expr = {node: 2 * spins[node] - 1 for node in graph.nodes()}
    
    # Set objective: minimize sum_uv J_uv Z_u Z_v + sum_i h_i Z_i
    objective = gp.quicksum(graph[u][v]['weight'] * spin_expr[u] * spin_expr[v] for u, v in graph.edges())
    objective += gp.quicksum(graph.nodes[node]['weight'] * spin_expr[node] for node in graph.nodes())
    
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Set time limit if provided
    if tlimit is not None:
        model.setParam('TimeLimit', tlimit)
    
    # Set the maximum number of cores if provided
    if max_cores is not None:
        model.setParam('Threads', max_cores)
    
    # Optimize the model
    model.optimize()
    
    # Extract the optimal cost and MIP gap
    cost = model.ObjVal
    mipgap = model.MIPGap
    
    # Extract the solution and create the solution list
    solution = {node: int(spins[node].X) for node in graph.nodes()}
    solution_list = [2 * solution[node] - 1 for node in sorted(graph.nodes())]  # Convert to -1 and 1
    
    return cost, mipgap, solution_list