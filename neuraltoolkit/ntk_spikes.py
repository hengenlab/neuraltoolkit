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
import matplotlib.pyplot as plt
import seaborn as sns


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


def cell_isi_hist(time_s, start=False, end=False, isi_thresh=0.1,
                  nbins=101, lplot=1):

    '''
    For a view of how much this cell is like a "real" neuron,
        calculate the ISI distribution between 0 and 100 msec.

    Return a histogram of the interspike interval (ISI) distribution.
    This is typically used to evaluate whether a spike train exhibits
    a refractory period and is thus consistent with a
    single unit or a multi-unit recording.
    This function will plot the bar histogram of that distribution
    and calculate the percentage of ISIs that fall under 2 msec.

    cell_isi_hist(time_s, start=False, end=False, isi_thresh=0.1,
                  nbins=101, lplot=1)

    Parameters
    ----------
    time_s : time in seconds
    start : Start time (default False uses 0)
    end : End time (default False uses np.max(time_s))
    isi_thresh : isi threshold (default 0.1)
    nbins : Number of bins (default 101)
    lplot : To plot or not (default lplot=1, plot isi)

    Returns
    -------
    ISI : spike time difference (a[i+1] - a[i]) along axis
    edges, hist_isi : To plot later
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.bar(edges[1:]*1000-0.5, hist_isi[0], color='#0b559f')

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    cell_isi_hist(time_s, start=False, end=False,
                  isi_thresh=0.1, nbins=101, lplot=1)

    '''

    if start is False:
        start = 0
    if end is False:
        end = np.max(time_s)

    # Calulate isi
    idx_l = np.where(np.logical_and(time_s >= start, time_s <= end))[0]
    ISI = np.diff(time_s[idx_l])

    # plot histogram and calculate contamination
    edges = np.linspace(0, isi_thresh, nbins)
    hist_isi = np.histogram(ISI, edges)

    # Calculate contamination percentage
    contamination = 100*(sum(hist_isi[0][0:int((0.1/isi_thresh) *
                         (nbins-1)/50)])/sum(hist_isi[0]))
    contamination = round(contamination, 2)
    cont_text = 'Contamination is {} percent.' .format(contamination)

    if lplot:
        plt.ion()
        with sns.axes_style("white"):
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.bar(edges[1:]*1000-0.5, hist_isi[0], color='#0b559f')
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            ax.set_xlabel('ISI (ms)')
            ax.set_ylabel('Number of intervals')
            ax.text(30, 0.7*ax.get_ylim()[1], cont_text)
        sns.despine()
    return ISI, edges, hist_isi
