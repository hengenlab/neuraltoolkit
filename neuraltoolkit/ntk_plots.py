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


def plot_data(data, data_beg, data_end, channel=0):

    '''
    Plot data defaults to channel 0
    plot_data(data, data_beg, data_end, channel=0)
    '''

    import matplotlib.pyplot as plt

    # plot
    plt.plot(data[channel, data_beg:data_end])
    plt.grid()
    plt.show()


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
