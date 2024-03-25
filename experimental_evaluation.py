"""
This file contains the implementation of the experimental evaluation of the lower bounding functions (T and P).

Author:
- Artur Dandolini Pescador
"""

from dtw_functions import dtw_distance, lb_keogh, lb_yi, lb_kim
import numpy as np

def compute_T(dataset):
    """
    Compute the ratio T for each lower bounding function.

    T 

    Parameters:
    - dataset: the dataset of time series sequences
    
    Returns:
    - T_Yi: the ratio T for the LB_Yi function
    - T_Kim: the ratio T for the LB_Kim function
    - T_Keogh: the ratio T for the LB_Keogh function
    """
    num_sequences = len(dataset)
    num_comparisons = int(num_sequences * (num_sequences - 1) / 2)
    
    # Initialize arrays to store distances
    true_distances = np.zeros(num_comparisons)
    LB_Yi_distances = np.zeros(num_comparisons)
    LB_Kim_distances = np.zeros(num_comparisons)
    LB_Keogh_distances = np.zeros(num_comparisons)
    
    idx = 0
    for i in range(num_sequences):
        for j in range(i+1, num_sequences):
            # Compute true DTW distance
            true_distances[idx] = dtw_distance(dataset[i], dataset[j])
            
            # Compute lower bound distances
            LB_Yi_distances[idx] = lb_yi(dataset[i], dataset[j])
            LB_Kim_distances[idx] = lb_kim(dataset[i], dataset[j])
            LB_Keogh_distances[idx] = lb_keogh(dataset[i], dataset[j], r=3)
            
            idx += 1
    
    # Calculate the ratio T for each lower bounding function
    T_Yi = np.mean(np.minimum(1, LB_Yi_distances / true_distances))
    T_Kim = np.mean(np.minimum(1, LB_Kim_distances / true_distances))
    T_Keogh = np.mean(np.minimum(1, LB_Keogh_distances / true_distances))
    
    return T_Yi, T_Kim, T_Keogh

def perform_experiment_T(dataset, query_lengths, sample_size=50):
    T_results = {length: {'yi': [], 'kim': [], 'keogh': []} for length in query_lengths}

    for length in query_lengths:
        sequences = [sequence[:length] for sequence in dataset[np.random.choice(len(dataset), sample_size, replace=False)]]

        T_results[length]['yi'], T_results[length]['kim'], T_results[length]['keogh'] = compute_T(sequences)

    return T_results


def compute_P(query, dataset):
    """
    Compute the Pruning Power (P) for each method.
    """
    LB_Yi_pruned_count = 0
    LB_Kim_pruned_count = 0
    LB_Keogh_pruned_count = 0
    
    for query_sequence in query:

        other_sequences = np.delete(dataset, np.where(dataset == query_sequence)[0], axis=0)
        
        # Calculate the true DTW distance
        true_distances = [dtw_distance(query_sequence, other_sequence) for other_sequence in other_sequences]
        
        LB_Yi_distances = [lb_yi(query_sequence, other_sequence) for other_sequence in other_sequences]
        LB_Kim_distances = [lb_kim(query_sequence, other_sequence) for other_sequence in other_sequences]
        LB_Keogh_distances = [lb_keogh(query_sequence, other_sequence, r=3) for other_sequence in other_sequences]

        # Find the nearest match using the true DTW distance
        nearest_match = np.argmin(true_distances)

        # Find the nearest match using the LB_Yi distance
        nearest_match_Yi = np.argmin(LB_Yi_distances)
        nearest_match_Kim = np.argmin(LB_Kim_distances)
        nearest_match_Keogh = np.argmin(LB_Keogh_distances)

        # If the nearest match using the LB_Yi distance is the same as the nearest match using the true DTW distance
        # then we have pruned the search space
        if nearest_match_Yi == nearest_match:
            LB_Yi_pruned_count += 1
        if nearest_match_Kim == nearest_match:
            LB_Kim_pruned_count += 1
        if nearest_match_Keogh == nearest_match:
            LB_Keogh_pruned_count += 1

    # Calculate the Pruning Power (P) for each method
    P_Yi = LB_Yi_pruned_count / len(query)
    P_Kim = LB_Kim_pruned_count / len(query)
    P_Keogh = LB_Keogh_pruned_count / len(query)

    return P_Yi, P_Kim, P_Keogh