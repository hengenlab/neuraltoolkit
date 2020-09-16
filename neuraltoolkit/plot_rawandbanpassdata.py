#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to plot raw data and bandpassed data

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


'''


try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')
try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError('Run command : conda install numpy')
import neuraltoolkit as ntk

# Get filename
print("Enter filename ")
rawfile = str(input())

# Get DAQ type
print("Enter 1 if ecube or 2 for intan")
lecube = np.int8(input())
print(lecube)

# Get number of channels
print("Enter total number of probes: ")
number_of_probes = np.int16(eval(input()))

# Get number of channels
print("Enter total number of channels : ")
number_of_channels = np.int16(eval(input()))
print(number_of_channels)

if number_of_probes > 1:
    hs = []
    for i in range(number_of_probes):
        # Get number of channels
        print("Enter probe type (Ex. hs64) : ")
        hstype = input()
        print(hstype)
        hs.append(hstype)
else:
    # Get number of channels
    print("Enter probe type (Ex. hs64) : ")
    hs = input()
    print(hs)

print("hstype ", hs)

# Get number of seconds
print("Enter total number of seconds to plot : ")
nsec = np.int16(eval(input()))
print(nsec)

# Get nchs
print("Enter every n channel to plot")
print("For ex: 4, plots channels")
print("[ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]")
nchs = np.int16(eval(input()))
print(nchs)


pltbytet = input("Plot by tetrode? (y/n)")

ch_list = np.arange(0, number_of_channels, nchs)
print(ch_list)


# get data
if lecube == 1:
    tt, ddgc = ntk.load_raw_binary_gain_chmap_nsec(rawfile, number_of_channels,
                                                   hs, 25000, nsec,
                                                   number_of_probes)
elif lecube == 2:
    t, ddgc = ntk.load_intan_raw_gain_chanmap(rawfile, number_of_channels,
                                              hs, number_of_probes)
else:
    raise ValueError('Error please check input, DAQtype')


# bandpass data
bdgc = ntk.butter_bandpass(ddgc, 500, 7500, 25000, 3)

# plot
if pltbytet == 'n':
    plt.figure(1)
    # ntk.plot_data_chlist(ddgc, 0, 25000*5, l)

    for i in range(len(ch_list)):
        # print(i, " ", ch_list[i])
        ax = plt.subplot(len(ch_list), 1, i+1)
        plt.plot(ddgc[ch_list[i], :])
        plt.xticks([])
        # plt.yticks([])
        # plt.box(on=None)

    plt.figure(2)
    # ntk.plot_data_chlist(bdgc, 0, 25000*5, l)
    for i in range(len(ch_list)):
        # print(i, " ", ch_list[i])
        ax = plt.subplot(len(ch_list), 1, i+1)
        plt.plot(bdgc[ch_list[i], :])
        plt.xticks([])
        # plt.yticks([])
        # plt.box(on=None)
    plt.show()

elif pltbytet == 'y':
    plt.figure(1)
    for i in np.arange(0, number_of_channels, 4):
        ch_list_tet = ch_list[i:(i+4)]
        for j in range(len(ch_list_tet)):
            # print(i, " ", ch_list[i])
            ax = plt.subplot(len(ch_list_tet), 1, j+1)
            plt.plot(bdgc[ch_list_tet[j], :])
            plt.xticks([])
            # plt.yticks([])
            # plt.box(on=None)
            plt.title('Ch ' + str(i+j+1))
        plt.show()
