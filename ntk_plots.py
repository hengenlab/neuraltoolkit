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



