from dtw_functions import dtw_distance, lb_keogh
import numpy as np

def perform_experiment_T(dataset, lb_func, num_sequences, sequence_length, r=3):
    results = []
    # Generate indices for starting positions of sequences
    indices = np.random.choice(len(dataset) - sequence_length, num_sequences, replace=False)
    
    for i in range(num_sequences - 1):
        for j in range(i + 1, num_sequences):
            # Select two random contiguous sequences from the dataset
            seq1 = dataset[indices[i]:indices[i] + sequence_length]
            seq2 = dataset[indices[j]:indices[j] + sequence_length]

            # Calculate lower bounding distance and actual DTW distance
            if lb_func == lb_keogh:
                lb = lb_func(seq1, seq2, r)
            else:
                lb = lb_func(seq1, seq2)
            dtw_dist = dtw_distance(seq1, seq2)

            # Calculate the tightness of the lower bound
            T = lb / dtw_dist if dtw_dist != 0 else 0
            results.append(T)

    return np.mean(results)

    
# Perform the experiment for different query lengths
def perform_experiment_for_lengths(dataset, lb_func, num_sequences, lengths, step, r=3):
    avg_tightness = []

    for length in lengths:
        avg_tightness.append(perform_experiment_T(dataset, lb_func, num_sequences, length, r))

    return lengths, avg_tightness