# -*- coding: utf-8 -*-

'''
Channel maps for all the probes used in our lab
Hlab
Author :Kiran Bhaskaran-Nair
'''


def channel_map_data(data, number_of_channels, hstype):
    import numpy as np

    '''
    Apply channel mapping
    channel_map_data(data, number_of_channels, hstype)
    hstype : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan', 'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap' and linear
    '''

    channel_map = find_channel_map(hstype, number_of_channels)
    dc = data[channel_map, :]
    return dc


def find_channel_map(hstype, number_of_channels):
    import numpy as np

    '''
    Get channel map data
    find_channel_map(hstype, number_of_channels)
    hstype : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan', 'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap' and linear
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

    # KS Si probe
    elif hstype == 'Si_64_KS_chmap':
        chan_map = np.array([7,  45, 5,  56, 4,  48, 1,  62, 9,  53, 10, 42,
                             14, 59, 13, 39, 18, 49, 16, 36, 23, 44, 19, 33,
                             26, 40, 22, 30, 31, 35, 25, 27, 3,  51, 2,  63,
                             8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54,
                             21, 55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41,
                             38, 47, 32, 37]) - 1

    # KT Si probe
    elif hstype == 'Si_64_KT_T1_K2_chmap':
        chan_map = np.array([14, 59, 10, 42, 9,  53, 1,  62, 4,  48, 5,  56,
                             7,  45, 13, 39, 18, 49, 16, 36, 23, 44, 19, 33,
                             26, 40, 22, 30, 31, 35, 25, 27, 3,  51, 2,  63,
                             8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54,
                             21, 55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41,
                             38, 47, 32, 37]) - 1

    # Linear probe
    elif hstype == 'linear':
        chan_map = np.arange(0, number_of_channels, 1)

    else:
        print("Error: Headstage type")
        raise ValueError('Headstage type not defined!')

    return chan_map
