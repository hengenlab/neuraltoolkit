#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions for plotting

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

List of functions/class in ntk_ecube
plot_data(data, data_beg, data_end, channel=0)
plot_data_chlist(data, data_beg, data_end, ch_list)
'''

import matplotlib.pyplot as plt
import seaborn as sns


def plot_data(data, data_beg, data_end, channel=0,
              sampling_rate=25000,
              file_name_with_path=None):

    '''
    Plot a specified range of data from a given channel.

    plot_data(data, data_beg, data_end, channel=0,
              sampling_rate=25000,
              file_name_with_path=None)

    Parameters:
    -----------
    data : np.ndarray
        The data to plot. Can be raw, bandpassed, or LFP data.
        Expected shape: (channels, samples).
    data_beg : int
        Starting sample index for the plot.
    data_end : int
        Ending sample index for the plot.
    channel : int, optional
        The channel index to plot (default is 0).
    sampling_rate : int, optional
        Sampling rate in Hz (default is 25000).
    file_name_with_path : str or None, optional
        If None, displays the plot.
        If a file path is provided
        (e.g., '/path/to/plot.png'), saves the plot to the file.

    Returns:
    --------
    None
    '''

    # plot
    # plt.plot(data[channel, data_beg:data_end])
    # plt.grid()
    # plt.show()

    color_list = ['#008080', '#ff7f50', '#a0db8e', '#b0e0e6']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))

    data = data[channel, data_beg:data_end]
    sns.lineplot(
        x=range(len(data)),
        y=data,
        ax=ax,
        color=color_list[3],
        # marker='o',
        linestyle='-',
        markersize=0.5,
        label=f'Ch{channel}',
        zorder=1.0
    )

    # Styling the plot
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlabel(f'Samples [Samples/{sampling_rate} = Seconds]', fontsize=12)
    ax.set_ylabel('Amplitude [\u03bcV]', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='major', pad=50)
    plt.tight_layout()
    if file_name_with_path is None:
        plt.show()
    else:
        plt.savefig(file_name_with_path)


def plot_data_chlist(data, data_beg, data_end, ch_list):

    '''
    plot data in range for channels in the list

    plot_data_chlist(data, data_beg, data_end, ch_list )
    data_beg, data_end : sample range
    ch_list : list of channels to plot

    l = np.array([5, 13, 31, 32, 42, 46, 47, 49, 51, 52, 53, 54 ])
    plot_data_chlist(data, 25000, 50000, l )
    '''

    import matplotlib.pyplot as plt

    for i in range(len(ch_list)):
        print(i, " ", ch_list[i])

        plt.subplot(len(ch_list), 1, i+1)
        plt.plot(data[ch_list[i], data_beg:data_end])

        plt.xticks([])
        plt.yticks([])
        plt.box(on=None)

    plt.show()
