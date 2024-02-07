import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None
        if len(self.adj_mat) == 0:
            raise ValueError("Cannot find minimum spanning tree for an empty graph")
        # For use with a priority queue/heap, change non-edges (edges of weight zero) to weight inf
        adj_mat = np.where(self.adj_mat==0, np.inf, self.adj_mat)
        
        # Use Prim's algorithm to construct a MST from self.adj_mat
        self.mst = np.zeros(adj_mat.shape)
        # Start with any node
        s = 0
        # Create an empty priority queue/heap
        pq = []
        # To avoid using _siftup/_siftdown, instead mark the "valid" elements of the heap:
        # {node: [cost of shortest path from S to node, (u in S, node)]}
        # The keys of the dictionary also serve as the unexplored set of nodes V
        pq_valid = {i:[adj_mat[s, i], (s, i)] 
                    for i in range(1, len(adj_mat))}
        # Insert unexplored nodes V, prioritized by the cost of the
        # cheapest known edge between v and S = {s}.
        for v, cost_v in pq_valid.items():
            heapq.heappush(pq, cost_v)
        # Continue drawing from heap until all nodes have been added to the MST (pq_valid is empty)
        while len(pq_valid) > 0:
            cost, (i, u) = heapq.heappop(pq)   # Remove cheapest unexplored node u
            # If a disconnected node is being explored, raise a ValueError
            if cost == np.inf:
                raise ValueError("Cannot find minimum spanning tree for disconnected graph")
            # If the node popped from the heap is invalid, try again
            if u not in pq_valid.keys():
                continue
            self.mst[i, u] = adj_mat[i, u]     # Add node u to the MST via the cheapest edge(S, u)
            self.mst[u, i] = adj_mat[u, i]
            pq_valid.pop(u)   # Removing node from set V is equivalent to adding node to set S
            # Decrease-key for nodes V, if the cost of the cheapest known
            # path between S and v has changed with the addition of u to S,
            # i.e., if edge(u, v) < cost(prev_S, v)
            for v, cost_v in pq_valid.items():
                if adj_mat[u, v] < cost_v[0]:
                    # Update cost(S, v) to edge(u, v) and (node in prev_S, v) to (u, v)
                    pq_valid[v] = [adj_mat[u, v], (u, v)]
                    heapq.heappush(pq, pq_valid[v])
