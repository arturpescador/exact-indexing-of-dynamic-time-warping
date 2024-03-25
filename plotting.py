"""
This file contains the implementation of the plotting functions.

Author:
- Artur Dandolini Pescador
"""

import matplotlib.pyplot as plt

def plot_timeseries(time_step, sunspots, figsize=(10, 6), label=None, Upper=None, Lower=None):
    plt.figure(figsize=figsize)
    plt.plot(time_step, sunspots, label=label, color='black', linestyle='--')
    if Upper is not None and Lower is not None:
        plt.plot(time_step, Upper, label='Upper Bound', color='red', linestyle='-')
        plt.plot(time_step, Lower, label='Lower Bound', color='red', linestyle='-')
        plt.fill_between(time_step, Upper, Lower, color='green', alpha=0.2, label='Bounds')
    plt.xlabel('Time')
    plt.ylabel('Sunspots')
    plt.title('Monthly Mean Total Sunspot Number')
    plt.legend()
    plt.show()

def plot_query_database_with_bounds(x, y, U, L):
    plt.figure(figsize=(14, 7))
    plt.plot(x, label='Candidate Sequence (C)', color='blue', linestyle='--')
    plt.plot(y, label='Original Sequence (Q)', color='black', linestyle='--')
    plt.plot(U, label='Upper Bound (U)', color='red')
    plt.plot(L, label='Lower Bound (L)', color='red')

    plt.fill_between(range(len(x)), L, U, color='green', alpha=0.2)
    plt.title('Query Sequence and Database Sequence with Upper and Lower Bounds')
    plt.xlabel('Time')
    plt.ylabel('Sunspot Number')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sequence_with_paa(sequence, paa, label):

    plt.figure(figsize=(14, 7))
    plt.plot(sequence, label=f'{label} Sequence', color='black', linestyle='--')
    plt.plot(paa, label=f'PAA ({label}, dim=10)', color='red')
    plt.title(f'{label} Sequence and Piecewise Aggregate Approximation (PAA)')
    plt.xlabel('Time')
    plt.ylabel('Sunspot Number')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sequence_with_bounds_and_paa(original, U, L, U_hat, L_hat):

    plt.figure(figsize=(14, 7))
    plt.plot(original, label='Original Sequence (Q)', color='black', linestyle='--')
    plt.plot(U_hat, label='Upper Bound (U_hat)', color='green')
    plt.plot(L_hat, label='Lower Bound (L_hat)', color='green')
    plt.plot(U, label='Original Upper Bound (U)', color='red', alpha=0.5)
    plt.plot(L, label='Original Lower Bound (L)', color='red', alpha=0.5)
    plt.legend()
    plt.title('Original Sequence with Bounds and PAA')
    plt.xlabel('Time')
    plt.ylabel('Sunspot Number')
    plt.grid(True)
    plt.show()

def plot_matches_knn_search(query_sequence, knn_search_results):
    plt.figure(figsize=(12, 6))

    # Plot the query sequence
    plt.plot(query_sequence, label='Query Sequence', color='black', linewidth=2)

    # Plot each matching sequence from the K-NN search results
    for i, (sequence, distance) in enumerate(knn_search_results):
        plt.plot(sequence, label=f'Match {i+1} (Distance: {distance:.2f})')

    plt.title('K-NN Search Results')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def plot_range_search_results(query_sequence, range_search_results, epsilon=10):
    plt.figure(figsize=(14, 7))

    # Plot the query sequence
    plt.plot(query_sequence, label='Query Sequence', color='black', linewidth=5)

    # Plot each sequence from the range search results
    for i, sequence in enumerate(range_search_results, start=1):
        plt.plot(sequence, label=f'Match {i}')

    plt.title(f'Range Search Results (±{epsilon} range)')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def plot_tighness(lengths, avg_tightness_keogh, avg_tightness_kim, avg_tightness_yi):
    plt.figure(figsize=(10, 6))
    plt.plot(lengths, avg_tightness_keogh, label='Keogh', marker='o')
    plt.plot(lengths, avg_tightness_kim, label='Kim', marker='x')
    plt.plot(lengths, avg_tightness_yi, label='Yi', marker='s')
    plt.xlabel('Query Length')
    plt.ylabel('Average Tightness of Lower Bound (T)')
    plt.title('Tightness of Lower Bound vs Query Length')
    plt.legend()
    plt.grid(True)
    plt.show()