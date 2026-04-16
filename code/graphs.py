"""
==============================================================================

Title:             Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models
Subtitle:          Graph generation and graph I/O utilities
Repository:        https://github.com/vijeycreative/NOpt_QAOA1_Tuning
Version:           1.0.0
Date:              16/04/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides:
  • Random graph generators for several benchmark families, including
    bipartite graphs, Sherrington-Kirkpatrick graphs, d-regular graphs,
    and Erdős-Rényi graphs.
  • Support for assigning weighted edges and optional node fields using
    either fixed ±1 weights or nonzero integer weights drawn from Gaussian
    distributions.
  • Utilities for computing simple graph statistics such as the total edge
    weight of a weighted graph.
  • Serialization helpers for storing graphs together with solution metadata
    in compressed JSON format and reading them back into NetworkX objects.
  • A helper routine for converting a graph with both node and edge weights
    into an equivalent graph containing edge weights only, by introducing an
    auxiliary node.

Implementation Notes
--------------------
• Randomness and reproducibility:
  Most graph generators accept a seed argument to allow reproducible instance
  generation. If a seed is not supplied, random seeds may be generated
  internally.

• Edge and node weights:
  When Gaussian-distributed weights are requested, this module ensures that
  sampled weights are nonzero by resampling until a nonzero integer is drawn.

• Graph storage:
  Graphs are serialized using NetworkX node-link format and stored in
  compressed JSON files together with auxiliary metadata such as cost,
  mipgap, and solution strings.

How to Cite
-----------
If you use this code in academic work, please cite:
  Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models,
  https://arxiv.org/abs/2501.16419
License
-------
MIT License © 2026 V. Vijendran

==============================================================================
"""

import json
import gzip
import random
import numpy as np
import networkx as nx


def generate_bipartite_graph(n, m, p, e_dist, seed=42):
    """
    Generate a connected random bipartite graph with weighted edges.

    This routine repeatedly samples a bipartite Erdős-Rényi graph until a
    connected instance is obtained. Edge weights are then assigned either as
    nonzero integers drawn from a Gaussian distribution or as unit weights.

    Parameters
    ----------
    n : int
        Number of nodes in the first bipartite partition.
    m : int
        Number of nodes in the second bipartite partition.
    p : float
        Probability of an edge between nodes in opposite partitions.
    e_dist : tuple or None
        Tuple ``(mean, variance)`` specifying the Gaussian distribution used
        to assign nonzero integer edge weights. If `None`, all edges are
        assigned weight 1.
    seed : int or None, optional
        Random seed for reproducibility. If `None`, a random seed is generated
        internally.

    Returns
    -------
    nx.Graph
        A connected weighted bipartite graph.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        # Generate a random seed
        random_seed = np.random.randint(0, 2**32 - 1)
        # Set the generated seed
        np.random.seed(random_seed)

    # Loop until a connected graph is generated
    G = None
    while G is None or not nx.is_connected(G):
        # Generate a random seed
        random_seed = np.random.randint(0, 2**32 - 1)
        # Set the generated seed
        np.random.seed(random_seed)
        # Create a bipartite graph
        G = nx.bipartite.random_graph(n, m, p, seed=seed)

    # Assign Gaussian-distributed positive integer weights to edges
    if e_dist is not None:
        mean, variance = e_dist
        for (u, v) in G.edges():
            G[u][v]['weight'] = generate_nonzero_weight(mean, variance)
    else:
        for (u, v) in G.edges():
            G[u][v]['weight'] = 1

    return G


def compute_total_edge_weight(G):
    """
    Compute the total weight of all edges in a graph.

    Parameters
    ----------
    G : nx.Graph
        Weighted graph whose edge weights are stored in the ``"weight"``
        attribute.

    Returns
    -------
    int or float
        Sum of all edge weights in the graph.
    """
    total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
    return total_weight


def generate_sherrington_kirkpatrick_graph(n, seed, fields=False):
    """
    Generate a Sherrington-Kirkpatrick graph with random ±1 couplings.

    This routine constructs the complete graph on ``n`` nodes and assigns each
    edge a random weight in ``{+1, -1}``. Optionally, random local field terms
    in ``{+1, -1}`` are also assigned to the nodes.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    seed : int
        Random seed for reproducibility.
    fields : bool, optional
        If `True`, assign random node weights in ``{+1, -1}`` in addition to
        the edge weights.

    Returns
    -------
    nx.Graph
        A complete weighted graph representing an SK instance.

    Example
    -------
    >>> G = generate_sherrington_kirkpatrick_graph(5, seed=42, fields=True)
    >>> for (u, v, w) in G.edges(data='weight'):
    ...     print(f"Edge ({u}, {v}) has weight {w}")
    >>> for node, weight in G.nodes(data='weight'):
    ...     print(f"Node {node} has weight {weight}")
    """
    random.seed(seed)

    # Create a complete graph with n nodes
    G = nx.complete_graph(n)

    # Assign random weights (+1 or -1) to each edge
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.choice([-1, 1])

    # If fields is True, assign random weights (+1 or -1) to each node
    if fields:
        for node in G.nodes():
            G.nodes[node]['weight'] = random.choice([-1, 1])

    return G


def generate_nonzero_weight(mean, variance):
    """
    Generate a nonzero integer weight from a Gaussian distribution.

    The routine repeatedly samples from a normal distribution with the given
    mean and variance until the rounded integer value is nonzero.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution.
    variance : float
        Variance parameter used in the normal draw.

    Returns
    -------
    int
        Nonzero integer sampled from the specified Gaussian distribution.
    """
    weight = 0
    while weight == 0:
        weight = int(np.random.normal(mean, variance))
    return weight


def generate_d_regular_graph(n, d, seed=None, fields=False, e_dist=None, n_dist=None):
    """
    Generate a weighted d-regular graph with optional node fields.

    This routine constructs a random d-regular graph on ``n`` nodes, assigns
    edge weights either from a nonzero Gaussian integer distribution or from
    random ``{+1, -1}`` values, and optionally assigns node weights as local
    fields.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    d : int
        Degree of each node.
    seed : int or None, optional
        Random seed for reproducibility.
    fields : bool, optional
        If `True`, assign weights to nodes as local fields.
    e_dist : tuple or None, optional
        Tuple ``(mean, variance)`` specifying the Gaussian distribution used
        to assign nonzero integer edge weights. If `None`, edges are assigned
        random weights in ``{+1, -1}``.
    n_dist : tuple or None, optional
        Tuple ``(mean, variance)`` specifying the Gaussian distribution used
        to assign nonzero integer node weights when `fields=True`. If `None`,
        nodes are assigned random weights in ``{+1, -1}``.

    Returns
    -------
    nx.Graph
        Weighted d-regular graph, optionally with node weights.

    Example
    -------
    >>> G = generate_d_regular_graph(5, 3, seed=42, fields=True, e_dist=(0, 1), n_dist=(0, 1))
    >>> for (u, v, w) in G.edges(data='weight'):
    ...     print(f"Edge ({u}, {v}) has weight {w}")
    >>> for node, weight in G.nodes(data='weight'):
    ...     print(f"Node {node} has weight {weight}")
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create a d-regular graph with n nodes
    G = nx.random_regular_graph(d, n, seed=seed)

    # Assign weights to edges
    if e_dist is not None:
        mean, variance = e_dist
        for (u, v) in G.edges():
            G[u][v]['weight'] = generate_nonzero_weight(mean, variance)
    else:
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.choice([-1, 1])  # 1

    # If fields is True, assign weights to nodes
    if fields:
        if n_dist is not None:
            mean, variance = n_dist
            for node in G.nodes():
                G.nodes[node]['weight'] = generate_nonzero_weight(mean, variance)
        else:
            for node in G.nodes():
                G.nodes[node]['weight'] = random.choice([-1, 1])

    return G


def generate_erdos_renyi_graph(n, p, seed=None, fields=False, e_dist=None, n_dist=None):
    """
    Generate a weighted Erdős-Rényi graph with optional node fields.

    This routine constructs an Erdős-Rényi random graph on ``n`` nodes with
    edge probability ``p``. Edge and node weights may be assigned either from
    nonzero Gaussian integer distributions or from default unit weights.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    p : float
        Probability of an edge between any pair of nodes.
    seed : int or None, optional
        Random seed for reproducibility.
    fields : bool, optional
        If `True`, assign node weights as local fields.
    e_dist : tuple or None, optional
        Tuple ``(mean, variance)`` specifying the Gaussian distribution used
        to assign nonzero integer edge weights. If `None`, edges are assigned
        weight 1.
    n_dist : tuple or None, optional
        Tuple ``(mean, variance)`` specifying the Gaussian distribution used
        to assign nonzero integer node weights when `fields=True`. If `None`,
        nodes are assigned weight 1.

    Returns
    -------
    nx.Graph
        Weighted Erdős-Rényi graph, optionally with node weights.

    Example
    -------
    >>> G = generate_erdos_renyi_graph(5, 0.5, seed=42, fields=True, e_dist=(0, 1), n_dist=(0, 1))
    >>> for (u, v, w) in G.edges(data='weight'):
    ...     print(f"Edge ({u}, {v}) has weight {w}")
    >>> for node, weight in G.nodes(data='weight'):
    ...     print(f"Node {node} has weight {weight}")
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create an Erdős-Rényi graph with n nodes and edge probability p
    G = nx.erdos_renyi_graph(n, p, seed=seed)

    # Assign weights to edges
    if e_dist is not None:
        mean, variance = e_dist
        for (u, v) in G.edges():
            G[u][v]['weight'] = generate_nonzero_weight(mean, variance)
    else:
        for (u, v) in G.edges():
            G[u][v]['weight'] = 1

    # If fields is True, assign weights to nodes
    if fields:
        if n_dist is not None:
            mean, variance = n_dist
            for node in G.nodes():
                G.nodes[node]['weight'] = generate_nonzero_weight(mean, variance)
        else:
            for node in G.nodes():
                G.nodes[node]['weight'] = 1

    return G


def write_graph_to_file(G, cost, mipgap, solution_string, filename):
    """
    Write a graph and associated solution metadata to a compressed JSON file.

    The graph is serialized using NetworkX node-link format. In addition to the
    graph itself, the file stores the solution cost, mipgap, and a solution
    string.

    Parameters
    ----------
    G : nx.Graph
        Graph to be written.
    cost : float
        Cost associated with the stored solution.
    mipgap : float
        MIP gap associated with the stored solution.
    solution_string : str
        String representation of the solution.
    filename : str
        Output filename. The file is written using gzip compression.

    Returns
    -------
    None
    """
    # Convert the graph to a dictionary
    graph_data = nx.node_link_data(G)

    # Add cost, mipgap, and solution string to the dictionary
    graph_data['cost'] = cost
    graph_data['mipgap'] = mipgap
    graph_data['solution_string'] = solution_string

    # Write the dictionary to a compressed JSON file
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=4)


def read_graph_from_file(filename):
    """
    Read a graph and associated solution metadata from a compressed JSON file.

    This routine reverses the serialization performed by `write_graph_to_file`,
    reconstructing the NetworkX graph and returning the stored solution
    metadata.

    Parameters
    ----------
    filename : str
        Name of the gzip-compressed JSON file to read.

    Returns
    -------
    (nx.Graph, float, float, str)
        Tuple ``(G, cost, mipgap, solution_string)`` containing the graph and
        the associated stored metadata.
    """
    # Read the dictionary from the compressed JSON file
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Extract the graph and additional data from the dictionary
    G = nx.node_link_graph(graph_data)
    cost = graph_data['cost']
    mipgap = graph_data['mipgap']
    solution_string = graph_data['solution_string']

    return G, cost, mipgap, solution_string


def convert_to_edge_weight_only_graph(G):
    """
    Convert a graph with node and edge weights into an equivalent edge-only graph.

    This helper removes node weights from the original graph and introduces a
    new auxiliary node. Each original node with a node weight is then connected
    to the auxiliary node with an edge carrying that weight. This allows local
    field terms to be represented using edge weights only.

    Parameters
    ----------
    G : nx.Graph
        Original graph with edge weights and optional node weights stored in
        the ``"weight"`` node attribute.

    Returns
    -------
    nx.Graph
        Equivalent graph containing only edge weights.
    """
    # Create a copy of the original graph without node weights
    new_G = G.copy()
    for node in new_G.nodes():
        if 'weight' in new_G.nodes[node]:
            del new_G.nodes[node]['weight']

    # Add a new node to the graph
    new_node = max(new_G.nodes()) + 1
    new_G.add_node(new_node)

    # Connect the new node to every other node with edge weights equal to the original node weights
    for node in G.nodes():
        if 'weight' in G.nodes[node]:
            new_G.add_edge(new_node, node, weight=G.nodes[node]['weight'])

    return new_G