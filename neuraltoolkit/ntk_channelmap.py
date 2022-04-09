#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Channel maps for all the probes used in our lab

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

List of functions/class in ntk_channelmap
channel_map_data(data, number_of_channels, hstype, nprobes=1)
find_channel_map(hstype, number_of_channels)
create_chanmap_file_for_oe()
'''


from __future__ import print_function
import numpy as np


def channel_map_data(data, number_of_channels, hstype, nprobes=1):

    '''
    Apply channel mapping
    channel_map_data(data, number_of_channels, hstype, nprobes=1)
    hstype : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan', 'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap' and linear
    nprobes : Number of probes (default 1)
    '''

    print(hstype)
    # Remove spaces, 'hs64 ' to 'hs64'
    hstype = list(map(str.strip, hstype))
    print(hstype)

    if nprobes == 1:
        channel_map = find_channel_map(hstype[0], number_of_channels)
    else:
        print("Number of probes", nprobes)
        # Get number of channels
        # restricted to symmetric probe
        print('Assuming all probes have same number of channels')
        nchannels_probe = np.int16(number_of_channels/nprobes)
        print("Number of channels per probe", nchannels_probe)
        for i in range(nprobes):

            # probe type
            hstype_probe = hstype[i]
            print(hstype_probe)

            chan_map = find_channel_map(hstype_probe, nchannels_probe)
            chan_map = chan_map + 1
            # print(chan_map)
            if i == 0:
                chan_mapt = chan_map
                if nprobes == i+1:
                    break
            else:
                chan_map = chan_map + chan_mapt.size
                chan_mapt = np.concatenate((chan_mapt, chan_map), axis=0)

            channel_map = chan_mapt - 1

    dc = data[channel_map, :]
    return dc


def find_channel_map(hstype, number_of_channels=None):

    '''
    Get channel map data
    find_channel_map(hstype, number_of_channels=None)
    hstype : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan32', 'intan16test2',

             'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap'
             Si_64_KS_chmap includes  8-K2., 5-KS., 1A-K2. probe
             Si_64_KT_T1_K2_chmap includes 5-KT. and  5-K2. probe

             'UCLA_Si1'
             'PCB_tetrode', 'EAB50chmap_00',

             'APT_PCB'
             and 'linear'

    number_of_channels default(None). For hstype 'linear' provide
             number_of_channels, usually 64 and 32
    '''

    # Ecube HS-64
    if hstype == 'hs64':
        chan_map = np.array([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,  4,
                             28, 32, 24, 20, 48, 44, 36, 40, 64, 60, 52, 56,
                             54, 50, 42, 46, 62, 58, 34, 38, 39, 35, 59, 63,
                             47, 43, 51, 55, 53, 49, 57, 61, 37, 33, 41, 45,
                             17, 21, 29, 25, 1,  5,  13, 9,  11, 15, 23, 19,
                             3,  7,  31, 27]) - 1

    # Ecube eibless-hs64_port32
    elif hstype == 'eibless-hs64_port32':
        chan_map = np.array([1,  5,  9,  13, 3,  7,  11, 15, 17, 21, 25, 29,
                             19, 23, 27, 31, 33, 37, 41, 45, 35, 39, 43, 47,
                             49, 53, 57, 61, 51, 55, 59, 63, 2,  6,  10, 14,
                             4,  8,  12, 16, 18, 22, 26, 30, 20, 24, 28, 32,
                             34, 38, 42, 46, 36, 40, 44, 48, 50, 54, 58, 62,
                             52, 56, 60, 64]) - 1

    # Ecube eibless-hs64_port64
    elif hstype == 'eibless-hs64_port64':
        chan_map = np.array([1,  5,  3,  7,  9,  13, 11, 15, 17, 21, 19, 23,
                             25, 29, 27, 31, 33, 37, 35, 39, 41, 45, 43, 47,
                             49, 53, 51, 55, 57, 61, 59, 63, 2,  6,  4,  8,
                             10, 14, 12, 16, 18, 22, 20, 24, 26, 30, 28, 32,
                             34, 38, 36, 40, 42, 46, 44, 48, 50, 54, 52, 56,
                             58, 62, 60, 64]) - 1

    # Intan 32
    elif hstype == 'intan32':
        chan_map = np.array([25, 26, 27, 28, 29, 30, 31, 32, 1,  2,  3,  4,
                             5,  6,  7,  8,  24, 23, 22, 21, 20, 19, 18, 17,
                             16, 15, 14, 13, 12, 11, 10, 9]) - 1
    # Intan 16 Test
    elif hstype == 'intan16test2':
        chan_map = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12,
                             13,  14,  15,  16]) - 1

    # KS Si probe
    # 8-K2.  stag. 2 col regular site size
    # 5-KS.  linear variable site size
    # 1A-K2. stag. 2 col regular site size
    elif hstype == 'Si_64_KS_chmap':
        chan_map = np.array([7,  45, 5,  56, 4,  48, 1,  62, 9,  53, 10, 42,
                             14, 59, 13, 39, 18, 49, 16, 36, 23, 44, 19, 33,
                             26, 40, 22, 30, 31, 35, 25, 27, 3,  51, 2,  63,
                             8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54,
                             21, 55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41,
                             38, 47, 32, 37]) - 1

    # KT Si probe
    # 5-KT. stag. 2 col, small site size
    # 5-K2. stag. 2 col, regular site
    elif hstype == 'Si_64_KT_T1_K2_chmap':
        chan_map = np.array([14, 59, 10, 42, 9,  53, 1,  62, 4,  48, 5,  56,
                             7,  45, 13, 39, 18, 49, 16, 36, 23, 44, 19, 33,
                             26, 40, 22, 30, 31, 35, 25, 27, 3,  51, 2,  63,
                             8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54,
                             21, 55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41,
                             38, 47, 32, 37]) - 1

    elif hstype == 'PCB_tetrode':
        chan_map = np.array([2, 41, 50, 62, 6, 39, 42, 47, 34, 44, 51, 56,
                             38, 48, 59, 64, 35, 53, 3, 37, 54, 57, 40, 43,
                             45, 61, 46, 49, 36, 33, 52, 55, 15, 5, 58, 60,
                             18, 9, 63, 1, 32, 14, 4, 7, 26, 20, 10, 13, 19,
                             22, 16, 8, 28, 25, 12, 17, 23, 29, 27, 21, 11,
                             31, 30, 24]) - 1

    elif hstype == 'EAB50chmap_00':
        chan_map = np.array([2,  4, 20, 35, 3, 19, 22, 32, 5, 15, 26, 31,
                             6,  9, 14, 38, 7, 10, 21, 24,  8, 17, 29, 34,
                             12, 13, 16, 28, 25, 27, 37, 47, 36, 39, 46,
                             64, 40, 48, 51, 54, 42, 45, 52, 58, 43, 56,
                             62, 63, 44, 49, 57, 60, 53, 55, 59, 61, 1,
                             11, 18, 23, 30, 33, 41, 50]) - 1

    elif hstype == 'APT_PCB':
        chan_map = np.array([2, 5, 1, 22, 9, 14, 18, 47, 23, 26, 31, 3, 35, 4,
                             7, 16, 34, 21, 12, 10, 29, 17, 8, 13, 11, 6, 38,
                             19, 24, 20, 15, 25, 37, 32, 28, 27, 52, 46, 41,
                             30, 61, 57, 54, 33, 55, 43, 63, 36, 58, 51, 60,
                             42, 40, 50, 64, 48, 59, 49, 44, 45, 62, 56, 53,
                             39]) - 1

    elif hstype == 'UCLA_Si1':
        chan_map = np.array([47, 43, 35, 50, 37, 55, 40, 58, 41, 60, 44, 64,
                             46, 51, 49, 63, 27, 61, 30, 57, 33, 36, 52, 39,
                             59, 42, 45, 48, 53, 56, 62, 54, 15, 1, 5, 9, 4,
                             7, 10, 14, 13, 20, 16, 19, 11, 22, 6, 25, 2, 18,
                             3, 24, 8, 23, 12, 28, 17, 26, 21, 32, 29, 31, 34,
                             38]) - 1

    # Linear probe
    elif hstype == 'linear':
        # checks
        if number_of_channels is None:
            raise \
                ValueError('For hstype linear provide number_of_channels')
        chan_map = np.arange(0, number_of_channels, 1)

    else:
        print("Error: Headstage type")
        raise ValueError('Headstage type not defined!')

    return chan_map


def create_chanmap_file_for_oe():

    '''
    Create channel mapping file for Open Ephys
    This is more as a script so answer the questions
    probe type : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
                 'intan', 'Si_64_KS_chmap',
                 'Si_64_KT_T1_K2_chmap' and linear
    '''

    from neuraltoolkit import ntk_channelmap as ntkc

    # Get number of channels
    print("Enter total number of probes: ")
    number_of_probes = np.int16(eval(input()))

    chan_mapt = np.empty(shape=[0, 0], dtype=np.int64)
    for i in range(number_of_probes):

        # Get number of channels
        print("Enter total number of channels : ")
        number_of_channels = np.int16(eval(input()))
        print(number_of_channels)

        # Get number of channels
        print("Enter probe type (Ex. hs64) : ")
        hstype = input()
        print(hstype)

        chan_map = ntkc.find_channel_map(hstype, number_of_channels)
        chan_map = chan_map + 1
        if i == 0:
            chan_mapt = chan_map
            if number_of_probes == i+1:
                break
        else:
            chan_map = chan_map + chan_mapt.size
            chan_mapt = np.concatenate((chan_mapt, chan_map), axis=0)

    # Get filename
    print("Enter filename to save data: ")
    filename = input()
    print(filename)

    # write channel map
    fid = open(filename, 'w')

    print('{', file=fid)
    print('"0": {', file=fid)

    print('"mapping": [', file=fid)
    for ii in range(chan_mapt.size):
        if ii < chan_mapt.size - 1:
            print((chan_mapt[ii]), ',', file=fid)
        else:
            print((chan_mapt[ii]), file=fid)
    print('],', file=fid)

    print('"reference": [', file=fid)
    for ii in range(chan_mapt.size):
        if ii < chan_mapt.size - 1:
            print('-1', ',', file=fid)
        else:
            print('-1', file=fid)
    print('],', file=fid)

    print('"enabled": [', file=fid)
    for ii in range(chan_mapt.size):
        if ii < chan_mapt.size - 1:
            print('true', ',', file=fid)
        else:
            print('true', file=fid)
    print(']', file=fid)
    print('},', file=fid)

    print('"refs": {', file=fid)
    print('"channels": [', file=fid)
    print('-1,', file=fid)
    print('-1,', file=fid)
    print('-1,', file=fid)
    print('-1', file=fid)
    print(']', file=fid)
    print('},', file=fid)

    print('"recording": {', file=fid)
    print('"channels": [', file=fid)
    for ii in range(chan_mapt.size):
        if ii < chan_mapt.size - 1:
            print('false', ',', file=fid)
        else:
            print('false', file=fid)
    print(']', file=fid)
    print('}', file=fid)
    print('}', file=fid)

    fid.close()
