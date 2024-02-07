import pytest
import numpy as np
import copy
from mst import Graph
from sklearn.metrics import pairwise_distances

np.random.seed(42)

def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """
    # If graph G has V nodes, the MST of G must have V nodes and V-1 edges,
    assert mst.shape == adj_mat.shape, "MST has the wrong number of nodes"
    num_edges = np.where(np.tril(mst) != 0, 1, 0).sum()
    num_nodes = adj_mat.shape[0]
    assert num_edges == num_nodes - 1, "MST has the wrong number of edges"
    
    # the MST must be minimally connected,
    remove_edge = np.random.randint(num_nodes)
    disconnected = copy.deepcopy(mst)
    for j in range(num_nodes):
        if disconnected[remove_edge][j] != 0:
            # Removing any edge disconnects the graph
            disconnected[remove_edge][j] = 0
            disconnected[j][remove_edge] = 0
            break
    disconnected_explored = [0]
    explored_idx = 0
    while explored_idx < len(disconnected_explored):
        v = disconnected_explored[explored_idx]
        # Explore graph by following edges
        # Given symmetric nature of adjacency matrix, zero-out the symmetric edge entry
        neighbors = np.argwhere(disconnected[v] != 0)[:,0].tolist()
        for n in neighbors:
            disconnected[n, v] = 0
        disconnected_explored += neighbors
        explored_idx += 1
    # If graph is disconnected, not all nodes will be explored
    assert len(disconnected_explored) < num_nodes, "MST is not minimally connected"
    
    # and the MST must be maximally acyclic.
    add_edge = np.random.randint(num_nodes)
    cyclic = copy.deepcopy(mst)
    for j in range(num_nodes):
        if cyclic[add_edge][j] == 0:
            # Adding any edge makes the graph cyclic
            cyclic[add_edge][j] = 1
            cyclic[j][add_edge] = 1
            break
    cyclic_explored = [0]
    explored_idx = 0
    while explored_idx < len(cyclic_explored):
        v = cyclic_explored[explored_idx]
        # Explore graph by following edges
        # Given symmetric nature of adjacency matrix, zero-out the symmetric edge entry
        neighbors = np.argwhere(cyclic[v] != 0)[:,0].tolist()
        for n in neighbors:
            cyclic[n, v] = 0
            if n in cyclic_explored:
                explored_idx = np.inf
            cyclic_explored.append(n)
        explored_idx += 1
    # If graph is cyclic, explored_idx was set to np.inf
    assert explored_idx == np.inf, "MST is not maximally acyclic"
        
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    Uses my own example graph, with exactly one MST solution
    
    """
    g = Graph(np.array(
                [ [0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 8, 8, 1],
                  [0, 1, 0, 0, 0, 8],
                  [1, 8, 0, 0, 0, 0],
                  [1, 8, 0, 0, 0, 8],
                  [0, 1, 8, 0, 8, 0] ]
              ))
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 5)
    mst_truth = np.array(
                [ [0, 1, 0, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0] ]
              )
    assert np.array_equal(g.mst, mst_truth)


def test_mst_student_bad_graph():
    """
    
    TODO: Write at least one unit test for MST construction.
    Checks that construct_mst() raises a ValueError when the input graph 
    is disconnected or empty.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.adj_mat[0] = np.zeros(len(g.adj_mat[0]))  # Disconnect node 0
    with pytest.raises(ValueError) as excinfo:
        g.construct_mst()
    g.adj_mat = np.zeros(g.adj_mat.shape)  # Make graph empty
    with pytest.raises(ValueError) as excinfo:
        g.construct_mst()
