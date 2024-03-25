"""
This file contains the implementation of the indexed structure for time series indexing.

Author:
- Artur Dandolini Pescador
"""

import numpy as np

class Node:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf # boolean
        self.entries = []

    def add_entry(self, entry):
        """Add an entry to the node."""
        self.entries.append(entry)

class Entry:
    def __init__(self, paa_representation, original_sequence=None, mbr=None, lb_distance=np.inf):
        self.paa_representation = paa_representation # PAA
        self.original_sequence = original_sequence # Original time series sequence
        self.mbr = mbr # MBR

class IndexedStructure:
    def __init__(self, paa_size):
        self.root = Node(is_leaf=True)  # Assuming root is leaf node
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
    
    def insert(self, timeseries):
        """Insert a time series into the indexed structure."""
        paa_representation = self.paa(timeseries)
        entry = Entry(paa_representation, original_sequence=timeseries)
        entry.mbr = self.create_mbr(paa_representation)  # Create MBR during insertion
        self.root.add_entry(entry)

    def create_mbr(self, paa_representation):
        """Create a Minimum Bounding Rectangle (MBR) for a PAA representation."""
        mbr = np.zeros((self.paa_size, 2))
        for i in range(self.paa_size):
            mbr[i][0] = paa_representation[i]
            mbr[i][1] = paa_representation[i]
        return mbr
    
    def retrieve_full_sequence(self, paa_representation):
        """Retrieve the full sequence from the indexed structure."""
        for entry in self.root.entries:
            if np.array_equal(entry.paa_representation, paa_representation):
                return entry.original_sequence
        return None