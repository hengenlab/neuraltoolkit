# -*- coding: utf-8 -*-

'''
Functions for plotting
Hlab
Author :Kiran Bhaskaran-Nair

List of functions in ntk_plots

Plot one channel data in range
plot_data
help(plot_data)
'''


def plot_data(data, data_beg, data_end, channel=0):
    import matplotlib.pyplot as plt

    '''
    Plot data defaults to channel 0
    plot_data(data, data_beg, data_end, channel=0)
    '''

    # plot
    plt.plot(data[channel, data_beg:data_end])
    plt.grid()
    plt.show()


def plot_data_chlist(data, data_beg, data_end, ch_list):
    import matplotlib.pyplot as plt
    import numpy as np

    '''
    plot data in range for channels in the list

    plot_data_chlist(data, data_beg, data_end, ch_list )
    data_beg, data_end : sample range
    ch_list : list of channels to plot

    l = np.array([5, 13, 31, 32, 42, 46, 47, 49, 51, 52, 53, 54 ])
    plot_data_chlist(data, 25000, 50000, l )
    '''

    for i in range(len(ch_list)):
        print(i, " ", ch_list[i])

        ax = plt.subplot(len(ch_list), 1, i+1)
        plt.plot(data[ch_list[i], data_beg:data_end])

        plt.xticks([])
        plt.yticks([])
        plt.box(on=None)

    plt.show()
