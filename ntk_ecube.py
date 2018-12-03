# -*- coding: utf-8 -*-

'''
Script to read ecube binary file 
Hlab
Author :Kiran Bhaskaran-Nair

List of functions in ntk_ecube 

Load raw data without gain or channel mapping
load_raw_binary
help(load_raw_binary)

Load raw data apply gain  no channel mapping
help(load_raw_binary_gain)

Load raw data apply gain and channel mapping
help(load_raw_binary_gain_chmap)


'''


# Load Ecube HS data
def load_raw_binary(name, number_of_channels):
    import numpy as np

    '''
    load ecube data
    load_raw_binary(name, number_of_channels)
    returns first timestamp and data
    '''

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype = np.uint64, count =  1)
    dr = np.fromfile(f, dtype = np.int16,  count = -1)
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    f.close()
    return tr, drr



def load_raw_binary_gain(name, number_of_channels):
    import numpy as np

    '''
    load ecube data and multiply gain
    load_raw_binary_gain(name, number_of_channels)
    returns first timestamp and data
    '''

    # constants
    gain = np.float64(0.19073486328125)

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype = np.uint64, count =  1)
    dr = np.fromfile(f, dtype = np.int16,  count = -1)
    f.close()
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    dg = drr*gain
    return tr, dg



def load_raw_binary_gain_chmap(name, number_of_channels, hstype):
    import numpy as np
    from neuraltoolkit import ntk_channelmap as ntkc

    '''
    load ecube data and multiply gain and apply channel mapping
    load_raw_binary_gain_chmap(name, number_of_channels, hstype)
    hstype : 'hs64', 'Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap' and linear
    returns first timestamp and data
    '''

    # constants
    gain = np.float64(0.19073486328125)

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype = np.uint64, count =  1)
    dr = np.fromfile(f, dtype = np.int16,  count = -1)
    f.close()
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    dg = drr*gain
    dgc = ntkc.channel_map_data(dg, number_of_channels, 'hs64')
    return tr, dgc




