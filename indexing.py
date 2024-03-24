"""
This file contains the implementation of the k-NN search and range search algorithms for time series indexing using the DTW distance.

Author:
- Artur Dandolini Pescador
"""

import heapq
from dtw_functions import lb_paa, mindist, dtw_distance
import numpy as np
from dtw_functions import paa


def knn_search(indexed_structure, query_sequence, k, U_hat, L_hat):
    query_paa = indexed_structure.paa(query_sequence)
    k_neighbors = []

    # Priority queue for nodes to visit
    queue = []
    heapq.heappush(queue, (0, indexed_structure.root))
    
    while queue and len(k_neighbors) < k:
        _, node = heapq.heappop(queue)
        
        if node.is_leaf:
            # Calculate LB_PAA for all sequences in the leaf and update nearest neighbors
            for entry in node.entries:
                lb_distance = lb_paa(entry.paa_representation, U_hat, L_hat)
                if lb_distance < np.inf:  # Changed from comparing with 'distance'
                    heapq.heappush(k_neighbors, (lb_distance, entry))
                    if len(k_neighbors) > k:
                        heapq.heappop(k_neighbors)
        else:
            # Calculate mindist for all MBRs in the node and update the priority queue
            for entry in node.entries:
                mindistance = mindist(query_paa, entry.mbr)
                heapq.heappush(queue, (mindistance, entry))
    
    # Make sure to sort the k_neighbors by distance before returning
    k_neighbors.sort(key=lambda x: x[0])
    
    # Get only the k nearest neighbors and their sequences
    return [(entry.original_sequence, dist) for dist, entry in k_neighbors[:k]]

def range_search(query_paa, epsilon, node, query_sequence, U_hat, L_hat):
    results = []

    if not node.is_leaf:
        # If it's a non-leaf node, recurse on its children.
        for child in node.entries:
            # mindist should be defined elsewhere to work with PAA representations
            if mindist(query_paa, child.mbr) <= epsilon:
                # child is actually another node, so we pass it on to the next recursion level
                results.extend(range_search(query_paa, epsilon, child, query_sequence, U_hat, L_hat))
    else:
        # If it's a leaf node, check all PAA points in the node.
        for entry in node.entries:
            # lb_paa should be defined elsewhere to calculate the lower-bounding distance using PAA
            if lb_paa(entry.paa_representation, U_hat, L_hat) <= epsilon:
                # dtw_distance should be defined elsewhere to calculate the actual DTW distance
                if dtw_distance(query_sequence, entry.original_sequence) <= epsilon:
                    results.append(entry.original_sequence)
    
    return results
"""

def range_search(query_paa, epsilon, node, query_sequence, U_hat, L_hat):
    results = []

    if not node.is_leaf:
        # If it's a non-leaf node, recurse on its children.
        for child in node.entries:
            if mindist(query_paa, child.mbr) <= epsilon:
                results.extend(range_search(query_paa, epsilon, child))
    else:
        # If it's a leaf node, check all PAA points in the node.
        for entry in node.entries:
            if lb_paa(entry.paa_representation, U_hat, L_hat) <= epsilon:
                # Retrieve full sequence from database if needed, for now, we use the stored one.
                if dtw_distance(query_sequence, entry.original_sequence) <= epsilon:
                    results.append(entry.original_sequence)
    
    return results
"""
