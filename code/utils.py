"""
==============================================================================
Title:             Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models
Subtitle:          Graph management and reduction utilities for RQAOA
Repository:        https://github.com/vijeycreative/NOpt_QAOA1_Tuning
Version:           1.0.0
Date:              16/04/2026

Author:            V Vijendran
Email:             vjqntm@gmail.com

Description
-----------
This module provides utilities for graph-based Recursive QAOA (RQAOA)
preprocessing and solution reconstruction. In particular, it includes:

  • A GraphManager class for performing in-place correlation and
    anti-correlation eliminations on weighted Ising graphs.
  • Bookkeeping utilities to track node mappings, eliminated variables,
    and spin-sign propagation required to reconstruct assignments on the
    original problem instance.
  • An exact brute-force solver for the residual reduced graph after
    elimination.
  • Helper routines for graph visualization, graph-to-array conversion,
    and extraction of edge/weight data for downstream numerical routines.
  • A utility to convert a graph with node and edge weights into an
    equivalent edge-only weighted graph by introducing an auxiliary node.

Implementation Notes
--------------------
• Graph mutation:
  The reduced graph is mutated in-place during variable elimination.
  All eliminations are recorded through node maps so that full assignments
  can be reconstructed at the end.

• External fields:
  If `fields_present=True`, node weights are interpreted as external fields
  in the Ising Hamiltonian and are propagated appropriately during elimination.

• Residual exact solve:
  The brute-force method is intended only for the small reduced instances
  that remain after RQAOA eliminations.

• Scope:
  This file focuses on correctness, transparency, and debugging convenience
  rather than aggressive optimization.

How to Cite
-----------
If you use this code in academic work, please cite:
  Near-Optimal Parameter Tuning of Level-1 QAOA for Ising Models -  https://arxiv.org/abs/2501.16419

License
-------
MIT License © 2026 V. Vijendran
==============================================================================
"""

import sys
import itertools
import numpy as np
from numba import jit
import networkx as nx
from scipy import optimize
from functools import partial
from numpy import sin, cos, pi
import matplotlib.pyplot as plt


def draw_graph(G):
    """
    Plot a weighted NetworkX graph using a circular layout.

    This helper is mainly intended for debugging intermediate RQAOA graph
    reductions. Positive-weight edges are drawn as solid lines, while
    negative-weight edges are drawn as dashed blue lines.

    Parameters
    ----------
    G : networkx.Graph
        Weighted graph whose edge weights are stored in the ``"weight"``
        attribute.

    Returns
    -------
    None
    """

    # Create a Matplotlib Figure and place the graph vertices in a circular layout
    plt.figure(figsize=(10,10))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=250)
    
    # Get all the edges with postive weights.
    epositive = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.0]
    # Get all the edges with negative weights.
    enegative = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.0]
    
    # Draw the positive and negatively weighted edges each with their own defined style.
    nx.draw_networkx_edges(G, pos, edgelist=epositive, width=3)
    nx.draw_networkx_edges(G, pos, edgelist=enegative, width=3, alpha=0.5, edge_color="b", style="dashed")

    # Draw the node labels on the figure.
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    # Get the edge weights and draw them on the figure.
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)


def draw_graph_with_fields(G):
    """
    Plot a weighted NetworkX graph including node-field values when present.

    Edges are styled in the same way as in ``draw_graph``. In addition, node
    weights stored in the ``"weight"`` node attribute are displayed in red near
    each node.

    Parameters
    ----------
    G : networkx.Graph
        Weighted graph with edge weights in the ``"weight"`` edge attribute and
        optional node weights in the ``"weight"`` node attribute.

    Returns
    -------
    None
    """

    # Create a Matplotlib Figure and place the graph vertices in a circular layout
    # plt.figure(figsize=(6,6))
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

    # Get all the edges with positive weights.
    epositive = [(u, v)
                 for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.0]
    # Get all the edges with negative weights.
    enegative = [(u, v)
                 for (u, v, d) in G.edges(data=True) if d["weight"] < 0.0]

    # Draw the positive and negatively weighted edges each with their own defined style.
    nx.draw_networkx_edges(G, pos, edgelist=epositive, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=enegative,
                           width=2, alpha=0.5, edge_color="b", style="dashed")

    # Draw the node labels on the figure.
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    # Get the edge weights and draw them on the figure.
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

    # Offset for the node weight labels
    label_offset = 0.05  # Adjust this value as needed
    pos_labels = {node: (coordinates[0] + label_offset, coordinates[1] + label_offset)
                  for node, coordinates in pos.items()}

    # Get the node weights and draw them in red with offset
    node_labels = nx.get_node_attributes(G, 'weight')
    nx.draw_networkx_labels(
        G, pos_labels, labels=node_labels, font_color='red')

    plt.show()


def has_edge(edge, edge_list):
    """
    Check whether an undirected edge appears in a list of edges.

    Since the graph is undirected, ``(u, v)`` and ``(v, u)`` are treated as
    equivalent.

    Parameters
    ----------
    edge : tuple
        Candidate edge of the form ``(u, v)``.
    edge_list : list[tuple]
        List of edges to search.

    Returns
    -------
    bool
        ``True`` if either ``edge`` or its reversed ordering is found in
        ``edge_list``; otherwise ``False``.
    """
    return (edge in edge_list) or (edge[::-1] in edge_list)


def graph_to_array(graph):
    """
    Convert a NetworkX graph into an edge array and adjacency matrix.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    Returns
    -------
    tuple
        A tuple ``(edges, adj_mat)`` where:

        - ``edges`` is a NumPy array containing the graph edges.
        - ``adj_mat`` is the weighted adjacency matrix of the graph, ordered
          according to sorted node labels.
    """
    node_list = list(graph.nodes())
    node_list.sort()
    adj_mat = nx.to_numpy_array(graph, node_list)
    
    return np.array(graph.edges()), adj_mat


def convert_to_edge_weight_only_graph(G):
    """
    Convert a graph with node and edge weights into an edge-only weighted graph.

    The transformation introduces a new auxiliary node connected to each
    original node with an edge weight equal to that node's original field
    weight. This is useful when one wants to represent node terms as edge
    terms in an equivalent extended graph.

    Parameters
    ----------
    G : networkx.Graph
        Original graph with edge weights and optional node weights stored in
        the ``"weight"`` node attribute.

    Returns
    -------
    networkx.Graph
        A new graph containing only edge weights.
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


class GraphManager:
    """
    Stateful manager for RQAOA variable elimination and reconstruction.

    The purpose of this class is three-fold:

    1. Store the original graph, which is used for evaluating the final cost.
    2. Maintain a reduced graph that is updated during each elimination step
       of RQAOA.
    3. Track node mappings so that assignments obtained on the reduced problem
       can be propagated back to the original variables.

    Attributes
    ----------
    original_graph : networkx.Graph
        Copy of the original input graph.
    reduced_graph : networkx.Graph
        Graph that is mutated during elimination.
    verbose : bool
        If ``True``, print elimination activity.
    fields_present : bool
        Whether node weights should be interpreted as external fields.
    nodes_vals : dict
        Dictionary mapping each original node to a spin value in ``{+1, -1}``.
    node_maps : dict
        Dictionary storing elimination mappings of the form
        ``eliminated_node -> (mapped_node, sign)``.
    remaining_nodes : list
        Nodes that have not yet been eliminated.
    """

    def __init__(self, graph, fields_present = False, verbose = False):
        """
        Initialize a GraphManager instance.

        Parameters
        ----------
        graph : networkx.Graph
            Input problem graph.
        fields_present : bool, optional
            Whether node weights should be treated as external fields.
        verbose : bool, optional
            If ``True``, print log messages during eliminations.
        """

        # This variable stores the original graph. No changes are made to this variable.
        self.original_graph = graph.copy()
        # This variable stores the graph after each variable elimination steps of the RQAOA.
        self.reduced_graph = graph
        # Boolean value to indicate whether or not to print activity.
        self.verbose = verbose
        # Boolean value to indicate whether the Ising Model has external fields present.
        self.fields_present = fields_present
        # This dictionary maps each node to a value in {1, -1}. This dictionary is used
        # to store the optimal assignments for the original problem.
        self.nodes_vals = {i: 0 for i in range(graph.number_of_nodes())}
        # This dictionary contains the mapping from one node to another based upon the
        # RQAOA's variable elimination method. Initially all nodes are mapped to themselves
        # with a +1 correlation.
        self.node_maps = {i: (i, 1) for i in range(graph.number_of_nodes())}
        # This list contains all the nodes that aren't eliminated yet. When the GraphManager
        # is first initialized, this list contains all nodes.
        self.remaining_nodes = [i for i in range(graph.number_of_nodes())]
        self.iter = 0
        self.optimal_angles = {}
        self.log = {}

    def correlate(self, edge):
        """
        Eliminate a node by enforcing positive correlation along an edge.

        Given an edge ``(u, v)``, this method imposes the constraint ``u = v``,
        removes node ``u`` from the reduced graph, and reattaches its incident
        edges to node ``v``. If external fields are present, node weights are
        merged accordingly.

        Parameters
        ----------
        edge : tuple
            Edge ``(u, v)`` in the reduced graph.

        Returns
        -------
        None
        """
        # Get the vertices u and v.
        u, v = edge
        # Make sure the reduced graph has the edge (u, v) or else you are trying to remove
        # something that is not there.
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."
        # d is the set of vertices other than v that are connected to u.
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)
        # Anti-Correlate by setting the map of u is v; i.e. we are eliminating the node u.
        self.node_maps[u] = (v, 1)
        self.remaining_nodes.remove(u)
        rm_msg1 = f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_msg1 + "\n"
        if self.verbose:
            print(rm_msg1)
        # Remove the edge (u, v) from the reduced graph.
        self.reduced_graph.remove_edge(v, u)
        # Get the weights of all the edges connected to the vertex 'u'.
        old_weights = {w: self.reduced_graph[w][u]['weight'] for w in d}

        # Iterate through all the neighbours of the vertex 'u' and remove their edges from the
        # reduced_graph.
        for w in d:
            rm_edge_msg = f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rm_edge_msg + "\n"
            if self.verbose:
                print(rm_edge_msg)
            self.reduced_graph.remove_edge(w, u)
        # Remove the vertex 'u' from the reduced_graph.
        rm_node_msg = f"Removing node {u} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)
        
        if self.fields_present:
            # Update Node Weights before Removing Node
            self.reduced_graph.nodes[v]['weight'] += self.reduced_graph.nodes[u]['weight']
        self.reduced_graph.remove_node(u)

        # Make new edges connecting the neighbours of the vertex 'u' to the vertex 'v'.
        # Use the old weights for the time being.
        new_edges = {(w, v): old_weights[w] for w in d}
        # Iterate thorugh all the new edges.
        for new_edge in new_edges:
            # If the reduced_graph already has the new_edge, then simply update the weights by
            # summing the existing_weight with the weight of the edge that was previously removed.
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        # Iterate through all the edges (w, v) and weight from the 'new_edges' dictionary.
        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                # If there are any edges with zero weight (that may have occurred to the previous loop),
                # then remove that edge from the reduced_graph.
                rm_new_edge_msg = f"Removing edge {new_edge} with weight {weight} from graph."
                self.log[self.iter] = self.log[self.iter] + rm_new_edge_msg + "\n"
                if self.verbose:
                    print(rm_new_edge_msg)
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                # Add all the edges with the non-zero weight to the reduced_graph datastructure.
                add_new_edge_msg = f"Adding edge {new_edge} with weight {weight} to graph."
                self.log[self.iter] = self.log[self.iter] + add_new_edge_msg + "\n"
                if self.verbose:
                    print(add_new_edge_msg)
                self.reduced_graph.add_edge(new_edge[0], new_edge[1], weight=weight)

    def anti_correlate(self, edge):
        """
        Eliminate a node by enforcing negative correlation along an edge.

        Given an edge ``(u, v)``, this method imposes the constraint ``u = -v``,
        removes node ``u`` from the reduced graph, and reattaches its incident
        edges to node ``v`` with the appropriate sign flip. If external fields
        are present, node weights are updated with the corresponding minus sign.

        Parameters
        ----------
        edge : tuple
            Edge ``(u, v)`` in the reduced graph.

        Returns
        -------
        None
        """
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."
        # d is the set of vertices other than v that are connected to u.
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)
        # Anti-Correlate by setting the map of u is v; i.e. we are eliminating the node u.
        self.node_maps[u] = (v, -1)
        self.remaining_nodes.remove(u)
        rm_msg1 = f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_msg1 + "\n"
        if self.verbose:
            print(rm_msg1)
        self.reduced_graph.remove_edge(v, u)

        old_weights = {w: self.reduced_graph[w][u]['weight'] for w in d}
        for w in d:
            rm_edge_msg = f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rm_edge_msg + "\n"
            if self.verbose:
                print(rm_edge_msg)
            self.reduced_graph.remove_edge(w, u)
        rm_node_msg = f"Removing node {u} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)

        if self.fields_present:
            # Update Node Weights before Removing Node
            self.reduced_graph.nodes[v]['weight'] += -1 * self.reduced_graph.nodes[u]['weight']
        self.reduced_graph.remove_node(u)
        new_edges = {(w, v): -old_weights[w] for w in d}

        for new_edge in new_edges:
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                rmv_new_edge_msg = f"Removing edge {new_edge} with weight {weight} from graph."
                self.log[self.iter] = self.log[self.iter] + rmv_new_edge_msg + "\n"
                if self.verbose:
                    print(rmv_new_edge_msg)
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                add_new_edge_msg = f"Adding edge {new_edge} with weight {weight} to graph."
                self.log[self.iter] = self.log[self.iter] + add_new_edge_msg + "\n"
                if self.verbose:
                    print(add_new_edge_msg)
                self.reduced_graph.add_edge(
                    new_edge[0], new_edge[1], weight=weight)

    def eliminate_node(self, node, sign):
        """
        Eliminate a node with a fixed spin assignment.

        This method is typically used when external fields are present. The
        chosen spin value is absorbed into neighbouring node fields, after
        which the node and its incident edges are removed.

        Parameters
        ----------
        node : int
            Node to eliminate.
        sign : int
            Assigned spin value, expected to be ``+1`` or ``-1``.

        Returns
        -------
        None
        """

        # node_neighbours is the set of vertices that are connected to node.
        neighbours = [w for w in self.reduced_graph[node]]
        for neighbour in neighbours:
            self.reduced_graph.nodes[neighbour]['weight'] += sign * \
                self.reduced_graph[node][neighbour]['weight']
            rmv_edge_msg = f"Removing edge {(node, neighbour)} with weight {self.reduced_graph[node][neighbour]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rmv_edge_msg + "\n"
            if self.verbose:
                print(rmv_edge_msg)
            self.reduced_graph.remove_edge(node, neighbour)

        rm_node_msg = f"Removing node {node} with weight {self.reduced_graph.nodes[node]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)

        self.nodes_vals[node] = sign
        self.remaining_nodes.remove(node)
        self.reduced_graph.remove_node(node)

    def get_root_node(self, node, s):
        """
        Recursively trace an eliminated node back to its root representative.

        Eliminated nodes may map to other eliminated nodes, which in turn map
        further until a non-eliminated root node is reached. This method follows
        that chain and accumulates the corresponding sign changes.

        Parameters
        ----------
        node : int
            Starting node whose root representative is sought.
        s : int
            Accumulated sign carried from previous mappings.

        Returns
        -------
        tuple
            A tuple ``(root_node, sign)`` where ``root_node`` is the final
            representative node and ``sign`` is the cumulative correlation sign.
        """
        # For a given 'node', 'mapped_tuple' is simply the tuple that contains new node
        # that 'node' is mapped to along with the + or - sign.
        mapped_tuple = self.node_maps[node]
        mapped_node, sign = mapped_tuple  # Unpacking the tuple.
        sign = sign * s  # Update the sign based the argument 's'
        # If the 'mapped_node' is indeed the root node then return it along with the sign,
        # else, recurse by calling get_root_node(mapped_node, sign).
        if self.fields_present:
            if (mapped_node in self.remaining_nodes) or (self.nodes_vals[mapped_node] != 0):
                return mapped_node, sign
            else:
                return self.get_root_node(mapped_node, sign)
        else:
            if (mapped_node in self.remaining_nodes):
                return mapped_node, sign
            else:
                return self.get_root_node(mapped_node, sign)

    def set_node_values(self, values):
        """
        Assign spin values to remaining nodes and propagate them to eliminated nodes.

        Parameters
        ----------
        values : list[int]
            List of spin assignments for the nodes in ``self.remaining_nodes``.
            Each entry must be either ``+1`` or ``-1``.

        Returns
        -------
        None
        """
        assert len(values) == len(self.remaining_nodes), "Number of values passed is not equal to the number of remaining nodes."
        for value in values:
            assert value == 1 or value == -1, "Values passed should be either 1 or -1."

        # Set the values of remaining set of nodes; i.e. the nodes that have been eliminated.
        for i, value in enumerate(values):
            node = self.remaining_nodes[i]
            self.nodes_vals[node] = value

        # Propagate the values of eliminated nodes based on the node mappings.
        for node, mapped_tuple in self.node_maps.items():
            mapped_node, sign = mapped_tuple
            # Skip Root Nodes (Only root nodes are mapped to themselves.)
            if node != mapped_node:
                # If map leads to a root node, apply the value.
                if mapped_node in self.remaining_nodes:
                    # The minus sign indicates that the two nodes are anti-correlated.
                    self.nodes_vals[node] = sign * self.nodes_vals[mapped_node]
                else:
                    root_node, s = self.get_root_node(mapped_node, sign)
                    self.nodes_vals[node] = s * self.nodes_vals[root_node]

    def compute_cost(self, graph):
        """
        Compute the Ising objective value for a graph under the current assignment.

        The cost is evaluated using the spin values stored in ``self.nodes_vals``.
        If ``fields_present=True``, node weights are also included as linear terms.

        Parameters
        ----------
        graph : networkx.Graph
            Graph on which the objective should be evaluated.

        Returns
        -------
        float
            Objective value for the current spin assignment.
        """
        # This for loop simply checks if the assignent values are either +1 or -1.
        for value in self.nodes_vals.values():
            assert value == 1 or value == -1, "All nodes should have a value of either 1 or -1."

        total_cost = 0
        for edge in graph.edges():
            total_cost += graph[edge[0]][edge[1]]['weight'] * self.nodes_vals[edge[0]] * self.nodes_vals[edge[1]]
        if self.fields_present:
            for node in graph.nodes():
                total_cost += graph.nodes[node]['weight'] * self.nodes_vals[node]

        return total_cost

    def brute_force(self):
        """
        Solve the residual reduced graph exactly by brute force.

        This method enumerates all assignments on the remaining nodes, selects
        the best reduced-graph assignment, and then reconstructs the full
        solution on the original graph.

        Returns
        -------
        tuple
            A tuple ``(best_cost, nodes_vals)`` where ``best_cost`` is the
            objective value on the original graph and ``nodes_vals`` is the
            reconstructed spin assignment dictionary.
        """

        num_values = len(self.remaining_nodes)
        assignments = list(map(list, itertools.product([1, -1], repeat=num_values)))

        best_reduced_cost = sys.maxsize
        best_assignment = assignments[0]

        for i, assignment in enumerate(assignments):
            self.set_node_values(assignment)
            reduced_cost = self.compute_cost(self.reduced_graph)

            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_assignment = assignment
                bf_msg = f"Best reduced cost found so far is {best_reduced_cost} for assignment {assignment}."
                self.log[self.iter] = self.log[self.iter] + bf_msg + "\n"
                if self.verbose:
                    print(bf_msg)

        self.set_node_values(best_assignment)
        best_cost = self.compute_cost(self.original_graph)

        best_cost_msg = f"Best Cost found for the original problem is {best_cost}."
        self.log[self.iter] = self.log[self.iter] + best_cost_msg + "\n"
        if self.verbose:
            print(best_cost_msg)

        return best_cost, self.nodes_vals
    

def extract_properties(graphmanager: GraphManager) -> tuple:
    """
    Extract the reduced edge list and full weight matrix from a GraphManager.

    The returned adjacency matrix is embedded in the indexing convention of the
    original graph, so eliminated nodes simply correspond to zero rows/columns
    unless diagonal field terms are present.

    Parameters
    ----------
    graphmanager : GraphManager
        GraphManager instance whose current reduced graph is to be extracted.

    Returns
    -------
    tuple
        A tuple ``(red_edges, adj_mat)`` where:

        - ``red_edges`` is a NumPy array containing the reduced graph edges.
        - ``adj_mat`` is a dense matrix containing the current reduced Ising
          couplings embedded in the original problem size. If external fields
          are present, diagonal entries store node weights.
    """

    red_edges, _ = graph_to_array(graphmanager.reduced_graph)
    num_nodes = graphmanager.original_graph.number_of_nodes()
    adj_mat = np.zeros((num_nodes, num_nodes))

    for u, v in red_edges:
        adj_mat[u][v] = graphmanager.reduced_graph[u][v]['weight']
        adj_mat[v][u] = graphmanager.reduced_graph[v][u]['weight']

    if graphmanager.fields_present:
        for i in graphmanager.reduced_graph.nodes:
            adj_mat[i][i] = graphmanager.reduced_graph.nodes[i]['weight']
    return red_edges, adj_mat