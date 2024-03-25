"""
This file contains the implementation of the k-NN search and range search algorithms for time series indexing using the DTW distance.

Author:
- Artur Dandolini Pescador
"""

import heapq
from dtw_functions import lb_paa, mindist, dtw_distance
import numpy as np
from indexed_structure import Entry

def knn_search(indexed_structure, query_sequence, k):
    """
    Perform a k-NN search on the indexed structure.

    Parameters:
    - indexed_structure: the indexed structure
    - query_sequence: the query sequence
    - k: the number of neighbors to retrieve

    Returns:
    - The k-nearest neighbors to the query sequence
    """
    # Init the MinPriorityQueue and the result list
    queue = []
    result = []

    # PAA of the query sequence
    query_paa = indexed_structure.paa(query_sequence)

    # Push the root node onto the queue with distance 0
    heapq.heappush(queue, (0, indexed_structure.root))

    while queue:
        top_distance, top_node = heapq.heappop(queue)

        # if top is a PAA point
        if isinstance(top_node, Entry):
            # Retrieve full sequence from the database
            full_sequence = indexed_structure.retrieve_full_sequence(top_node.paa_representation)

            # distance between the query sequence and the full sequence
            actual_distance = dtw_distance(query_sequence, full_sequence)
            result.append((full_sequence, actual_distance))

            if len(result) == k:  # If we have found k neighbors
                break

        elif top_node.is_leaf:  # if top is a leaf node
            for data_item in top_node.entries:
                # distance or the appropriate lower bound distance
                distance = dtw_distance(query_paa, data_item.paa_representation)
                heapq.heappush(queue, (distance, data_item))

        else:  # if top is a non-leaf node
            for child_node in top_node.entries:
                distance = mindist(query_paa, child_node.mbr)
                heapq.heappush(queue, (distance, child_node))

    # Sort the results by distance
    result.sort(key=lambda x: x[1])

    # Return only the k-nearest neighbors
    k_nearest_neighbors = result[:k]
    return k_nearest_neighbors

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