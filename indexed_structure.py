"""
This file contains the implementation of the indexed structure for time series indexing.

Author:
- Artur Dandolini Pescador
"""

import numpy as np
import heapq

class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)

class Entry:
    def __init__(self, paa_representation, original_sequence=None, mbr=None):
        self.paa_representation = paa_representation
        self.original_sequence = original_sequence
        self.mbr = mbr

class IndexedStructure:
    def __init__(self, paa_size):
        self.root = Node(is_leaf=True)  # Assuming root is leaf for simplicity
        self.paa_size = paa_size

    def paa(self, timeseries):
        """Piecewise Aggregate Approximation (PAA)"""
        n = len(timeseries)
        paa = np.zeros(self.paa_size)
        for i in range(self.paa_size):
            start = int(i * n / self.paa_size)
            end = int((i + 1) * n / self.paa_size)
            paa[i] = np.mean(timeseries[start:end])
        return paa
    
    #def create_paa_bounds(self, timeseries, bins, paa_size):
    #    paa_representation = self.paa(timeseries)
    #    U_hat = np.zeros(paa_size)
    #    L_hat = np.zeros(paa_size)
    #    for i in range(paa_size):
    #        start = max(0, i - bins)
    #        end = min(paa_size, i + bins)
    #        U_hat[i] = np.max(paa_representation[start:end])
    #        L_hat[i] = np.min(paa_representation[start:end])
    #    return U_hat, L_hat

    def insert(self, timeseries):
        paa_representation = self.paa(timeseries)
        entry = Entry(paa_representation, original_sequence=timeseries)
        entry.mbr = self.create_mbr(paa_representation)  # Create MBR during insertion
        self.root.add_entry(entry)

    def create_mbr(self, paa_representation):
        # Assuming a simple MBR where each dimension has the same fixed width
        mbr_width = 0.5  # This is a chosen parameter that defines the MBR width
        return [(paa - mbr_width, paa + mbr_width) for paa in paa_representation]