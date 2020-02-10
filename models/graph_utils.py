from copy import deepcopy

import graphviz
import numpy as np


def get_prev_nodes(matrix, node):
    """
    Returns the list of the nodes arriving at the given node
    :param matrix: adjacency matrix of the graph
    :param node: given node index
    :return:
    """
    n_nodes = matrix.shape[0]
    assert 0 <= node < n_nodes
    return np.where(matrix[:node, node])[0]


def get_next_nodes(matrix, node):
    """
    Returns the list of the nodes leaving the given node
    :param matrix: adjacency matrix of the graph
    :param node: given node index
    :return:
    """
    n_nodes = matrix.shape[0]
    assert 0 <= node < n_nodes
    return np.where(matrix[node, node:])[0] + node


def get_reaching_nodes(matrix, node):
    """
    Returns the list of the nodes that can reach the given node
    :param matrix: adjacency matrix of the graph
    :param node: given node
    """
    n_nodes = matrix.shape[0]
    assert 0 <= node < n_nodes

    visited = [0 for _ in range(n_nodes)]
    visited[node] = 1

    queue = list(get_prev_nodes(matrix, node))
    for q in queue:

        if not visited[q]:
            queue.extend(get_prev_nodes(matrix, q))
        visited[q] = 1

    return np.nonzero(visited)[0]


def get_reachable_nodes(matrix, node):
    """
    Returns the list of nodes that can be reached from the given node
    :param : adjacency matrix of the graph
    :param : given node
    """
    n_nodes = matrix.shape[0]
    visited = [0 for _ in range(n_nodes)]
    visited[node] = 1

    queue = list(get_next_nodes(matrix, node))
    for q in queue:

        if not visited[q]:
            queue.extend(get_next_nodes(matrix, q))
        visited[q] = 1

    return np.nonzero(visited)[0]


def get_n_active_nodes(matrix):
    """
    Returns the number of active nodes in the graph given its topology
    :param matrix:
    :return:
    """
    n_nodes = matrix.shape[0]
    reachable_input = get_reachable_nodes(matrix, 0)
    reaching_output = get_reaching_nodes(matrix, n_nodes - 1)
    return len(np.where([(i in reachable_input) and (i in reaching_output)
                         for i in range(n_nodes)]))


def simplify_matrix(matrix):
    """
    Remove useless edges from the matrix representation
    :return:
    """
    matrix = deepcopy(matrix)
    n_nodes = matrix.shape[0]
    reachable_input = get_reachable_nodes(matrix, 0)
    reaching_output = get_reaching_nodes(matrix, n_nodes - 1)

    for i in range(n_nodes):

        if not (i in reachable_input and i in reaching_output):
            matrix[i] = 0
            matrix[:, i] = 0

    return matrix


def print_graph(matrix, labels=None, show=False):
    """
    Displays the graph represented by the given adjacency matrix and node labels
    :param matrix:
    :param labels:
    """
    g = graphviz.Digraph(format='png', edge_attr=dict(fontsize='20', fontname="roman"),
                         node_attr=dict(style='filled', shape='circle', align='center', fontsize='20',
                                        height='0.5', width='0.5', penwidth='2', fontname="roman"),
                         engine='dot')
    g.body.extend(['rankdir=BT'])

    # create the nodes
    n_nodes = matrix.shape[0]
    for i in range(n_nodes):
        g.node(str(i),
               color="orangered" if i in [0, n_nodes - 1] else "orange",
               )

    # create the edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if matrix[i, j]:
                g.edge(str(i), str(j))

    # render graph
    g.render(view=show)

    return g


def get_all_next_nodes(matrix):
    """
    Returns the list of the leaving nodes
    :param matrix:
    :return:
    """
    num_vertices = np.shape(matrix)[0]
    next_nodes = [[] for _ in range(num_vertices)]

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if matrix[i, j]:
                next_nodes[i].append(j)

    return next_nodes


def get_all_prev_nodes(matrix):
    """
    Returns the list of the arriving nodes
    :param matrix:
    :return:
    """
    num_vertices = np.shape(matrix)[0]
    prev_nodes = [[] for _ in range(num_vertices)]

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if matrix[i, j]:
                prev_nodes[j].append(i)

    return prev_nodes


def compute_vertex_channels(input_channels, output_channels, matrix):
    """
    Adapted from https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py
    Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    :param input_channels: input channel count.
    :param output_channels: output channel count.
    :param matrix: adjacency matrix for the module (pruned by model_spec).
    :return: list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        # assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels
