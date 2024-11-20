#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Function to analyze spikes

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

'''

import numpy as np


def make_spikematrix(spiketimes_ms, start, end, binsz, binarize=False):
    """
    Create a spike matrix with counts or binary values based on spiking times.

    Parameters:
    - spiketimes_ms (list of lists): Spike times for each neuron (in ms).
    - start (float): Start time (ms).
    - end (float): End time (ms).
    - binsz (float): Bin size (ms).
    - binarize (bool): If True, binarize the spike counts (default: False).

    Returns:
    - np.ndarray: Matrix of shape (n_cells, n_bins)
        with spike counts or binary values.
    """

    # Define bin edges and initialize spike matrix
    bin_edges = np.arange(start, end + binsz, binsz)
    n_cells = len(spiketimes_ms)
    spike_matrix = np.zeros((n_cells, len(bin_edges) - 1))

    # Populate spike matrix
    for i, spiketimes in enumerate(spiketimes_ms):
        # Histogram counts for each cell
        counts, _ = np.histogram(spiketimes, bins=bin_edges)
        if binarize:
            counts = (counts > 0).astype(int)
        spike_matrix[i] = counts

    return spike_matrix
