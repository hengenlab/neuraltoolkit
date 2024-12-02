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


def plot_data_chlist(data, data_beg, data_end, ch_list=None,
                     sampling_rate=25000,
                     file_name_with_path=None,
                     title_string=None):
    """
    Plot a specified range of data from a given list of channels.

    Parameters:
    -----------
    data : np.ndarray
        The data to plot. Can be raw, bandpassed, or LFP data.
        Expected shape: (channels, samples).
    data_beg : int
        Starting sample index for the plot.
    data_end : int
        Ending sample index for the plot.
    ch_list : list
        The list of channel indices to plot.
    sampling_rate : int, optional
        Sampling rate in Hz (default is 25000).
    file_name_with_path : str or None, optional
        If None, displays the plot.
        If a file path is provided
        (e.g., '/path/to/plot.png'), saves the plot to the file.
    title_string : str or None, optional
        If None, not title to plot.
        If title_string is provided add it as title plot

    Returns:
    --------
    None
    """

    base_colors = ['#008080', '#ff7f50', '#a0db8e', '#b0e0e6', '#dda0dd',
                   '#f5deb3', '#808000', '#ffc0cb', '#ffa07a', '#20b2aa',
                   '#7fffd4', '#d8bfd8', '#da70d6', '#dda0dd', '#ffb6c1',
                   '#db7093', '#f0e68c', '#fafad2', '#ffdead', '#f5f5dc',
                   '#fff8dc', '#a52a2a', '#8b4513', '#deb887']
    needed_colors = len(ch_list)
    color_list = \
        (base_colors * (needed_colors // len(base_colors) + 1))[:needed_colors]

    fig, ax = plt.subplots(nrows=len(ch_list), ncols=1,
                           # sharex=True,
                           # sharey=True,
                           figsize=(16, 2 * len(ch_list)))

    ax[0].set_title(title_string)
    for indx, channel in enumerate(ch_list):
        data_ch = data[channel, data_beg:data_end]
        sns.lineplot(
            x=range(len(data_ch)),
            y=data_ch,
            ax=ax[indx],
            color=color_list[indx],
            linestyle='-',
            markersize=0.5,
            label=f'Ch{channel}',
            zorder=1.0
        )

        # Styling subplots
        ax[indx].legend(fontsize=6, loc='upper right')
        ax[indx].spines['top'].set_visible(False)
        ax[indx].spines['right'].set_visible(False)
        ax[indx].spines['left'].set_visible(False)
        ax[indx].spines['bottom'].set_visible(False)

        # Only show x-label for the last subplot
        if indx == (len(ch_list) - 1):
            ax[indx].set_xlabel(f'Samples [Samples/{sampling_rate} = Seconds]',
                                fontsize=10)
            ax[indx].set_ylabel('Amplitude [\u03bcV]', fontsize=10)
            ax[indx].tick_params(axis='x', which='both',
                                 bottom=True, top=True,
                                 labelbottom=True)

        else:
            ax[indx].set_xticklabels([])
            ax[indx].tick_params(axis='x', which='both',
                                 bottom=False, top=False,
                                 labelbottom=False)

    plt.tight_layout()

    if file_name_with_path is None:
        plt.show()
    else:
        plt.savefig(file_name_with_path)
