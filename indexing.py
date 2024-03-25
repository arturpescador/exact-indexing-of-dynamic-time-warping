"""
This file contains the implementation of the k-NN search and range search algorithms for time series indexing using the DTW distance.

Author:
- Artur Dandolini Pescador
"""

import heapq
from dtw_functions import lb_paa, mindist, dtw_distance
import numpy as np

def knn_search(indexed_structure, query_sequence, k, U_hat, L_hat):
    """
    Perform a k-NN search on the indexed structure using the DTW distance.

    Parameters:
    - indexed_structure: the indexed structure
    - query_sequence: the query sequence
    - k: the number of neighbors to retrieve
    - U_hat: the upper bound
    - L_hat: the lower bound

    Returns:
    - The k-nearest neighbors to the query sequence
    """
    query_paa = indexed_structure.paa(query_sequence)
    k_neighbors = []

    # Priority queue for nodes to visit
    queue = []

    # add the root node to the queue with distance 0
    heapq.heappush(queue, (0, indexed_structure.root))
    
    while queue and len(k_neighbors) < k:
        _, node = heapq.heappop(queue)
        
        if node.is_leaf:
            # Calculate LB_PAA
            for entry in node.entries:
                lb_distance = lb_paa(entry.paa_representation, U_hat, L_hat)
                if lb_distance < np.inf:
                    heapq.heappush(k_neighbors, (lb_distance, entry))
                    if len(k_neighbors) > k:
                        heapq.heappop(k_neighbors)
        else:
            # Calculate mindist for all MBRs in the node and update the priority queue
            for entry in node.entries:
                mindistance = mindist(query_paa, entry.mbr)
                heapq.heappush(queue, (mindistance, entry))
    
    # sort the k_neighbors by distance
    k_neighbors.sort(key=lambda x: x[0])
    
    return [(entry.original_sequence, dist) for dist, entry in k_neighbors[:k]]

def range_search(query_paa, epsilon, node, query_sequence, U_hat, L_hat):
    results = []

    if not node.is_leaf:
        for child in node.entries:
            if mindist(query_paa, child.mbr) <= epsilon:
                results.extend(range_search(query_paa, epsilon, child, query_sequence, U_hat, L_hat))
    else: # leaf node
        for entry in node.entries:
            if lb_paa(entry.paa_representation, U_hat, L_hat) <= epsilon:
                if dtw_distance(query_sequence, entry.original_sequence) <= epsilon:
                    results.append(entry.original_sequence)
    
    return results