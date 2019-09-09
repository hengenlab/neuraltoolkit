#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to read ecube binary file

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


List of functions/class in ntk_ecube
load_raw_binary(name, number_of_channels)
load_raw_binary_gain(name, number_of_channels)
load_raw_binary_gain_chmap(name, number_of_channels, hstype, nprobes=1,
                           t_only=0)
load_raw_binary_gain_chmap_nsec(name, number_of_channels, hstype,
                                fs, nsec, nprobes=1)
load_raw_binary_preprocessed(name, number_of_channels)
load_digital_binary(name, t_only=0)
'''

try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')


# Load Ecube HS data
def load_raw_binary(name, number_of_channels):

    '''
    load ecube data
    load_raw_binary(name, number_of_channels)
    returns first timestamp and data
    '''

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    dr = np.fromfile(f, dtype=np.int16,  count=-1)
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    f.close()
    return tr, drr


def load_raw_binary_gain(name, number_of_channels):

    '''
    load ecube data and multiply gain
    load_raw_binary_gain(name, number_of_channels)
    returns first timestamp and data
    '''

    # constants
    gain = np.float64(0.19073486328125)

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    dr = np.fromfile(f, dtype=np.int16,  count=-1)
    f.close()
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    dg = drr*gain
    return tr, dg


def load_raw_binary_gain_chmap(name, number_of_channels, hstype, nprobes=1,
                               t_only=0):

    '''
    load ecube data and multiply gain and apply channel mapping
    load_raw_binary_gain_chmap(name, number_of_channels, hstype)
    hstype : 'hs64', 'Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap' and linear
    nprobes : Number of probes (default 1)
    t_only  : if t_only=1, just return tr, timestamp
              (Default 0, returns timestamp and data)
    returns first timestamp and data
    '''

    from neuraltoolkit import ntk_channelmap as ntkc

    if isinstance(hstype, str):
        hstype = [hstype]

    assert len(hstype) == nprobes, \
            'length of hstype not same as nprobes'

    # constants
    gain = np.float64(0.19073486328125)

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    if t_only:
        f.close()
        return tr

    dr = np.fromfile(f, dtype=np.int16,  count=-1)
    f.close()
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    dg = drr*gain
    dgc = ntkc.channel_map_data(dg, number_of_channels, hstype, nprobes)
    return tr, dgc


def load_raw_binary_gain_chmap_nsec(name, number_of_channels, hstype,
                                    fs, nsec, nprobes=1):

    '''
    load ecube nsec of data and multiply gain and apply channel mapping
    load_raw_binary_gain_chmap_nsec(name, number_of_channels, hstype,
                                    fs, nsec):
    hstype : 'hs64', 'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan', 'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap' and linear
    fs : sampling rate
    nsec : number of seconds
    nprobes : Number of probes (default 1)
    returns first timestamp and data

    tt, ddgc = ntk.load_raw_binary_gain_chmap_nsec(rawfile, number_of_channels,
                                                   'hs64', 25000, 2)
    '''

    from neuraltoolkit import ntk_channelmap as ntkc

    if isinstance(hstype, str):
        hstype = [hstype]

    assert len(hstype) == nprobes, \
            'length of hstype not same as nprobes'

    # constants
    gain = np.float64(0.19073486328125)

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    dr = np.fromfile(f, dtype=np.int16,  count=nsec*number_of_channels*fs)
    f.close()
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    dg = drr*gain
    dgc = ntkc.channel_map_data(dg, number_of_channels, hstype, nprobes)
    return tr, dgc


# Load Ecube HS data preprocessed
def load_raw_binary_preprocessed(name, number_of_channels):

    '''
    load ecube data preprocessed
    load_raw_binary(name, number_of_channels)
    returns data
    '''

    f = open(name, 'rb')
    dr = np.fromfile(f, dtype=np.int16,  count=-1)
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    f.close()
    return drr


# Load Ecube HS data and grab 1 channel data
def load_a_ch(name, number_of_channels, channel_number,
              lraw=1, block_size=25000):

    '''
    load ecube data preprocessed and return a channels data
    load_a_ch(name, number_of_channels, channel_number, lraw, block_size)
    name - name of file
    number_of_channels - number of channels
    channel_number - channel data to return
    lraw - whether raw file or not  (default : raw lraw=1)
    block_size = block_size to read at a time
    returns data_of_channel_number
    '''

    gain = np.float64(0.19073486328125)
    dp_ch = np.array([])

    f = open(name, 'rb')

    if lraw:
        f_length = (f.seek(0, 2) - 8)/2/number_of_channels
        f.seek(8, 0)
    else:
        f_length = f.seek(0, 2)/2/number_of_channels
        f.seek(0, 0)
    nbs = np.ceil(f_length/block_size)

    for i in range(np.int64(nbs)):
        d = np.frombuffer(f.read(block_size*2*number_of_channels),
                          dtype=np.int16)
        dp_ch = np.append(dp_ch, d[channel_number:-1:number_of_channels])

    f.close()

    if lraw:
        dp_ch = np.int16(dp_ch * gain)

    return dp_ch


# Load Ecube HS data and grab data within range
def load_raw_binary_gain_chmap_range(rawfile, number_of_channels,
                                     hstype, nprobes=1,
                                     lraw=1, ts=0, te=25000):

    '''
    load ecube data preprocessed and return a channels data
    load_a_ch(name, number_of_channels, channel_number, lraw, block_size)
    name - name of file
    number_of_channels - number of channels
    channel_number - channel data to return
    hstype : Headstage type, 'hs64'
    nprobes : Number of probes (default 1)
    lraw - whether raw file or not  (default : raw lraw=1)
    ts : sample start (not seconds but samples),
           default : 0 ( begining of file)
    ts : sample end (not seconds but samples),
           default : -1 ( full data)
    returns time, data
    '''

    from neuraltoolkit import ntk_channelmap as ntkc

    gain = np.float64(0.19073486328125)
    d_bgc = np.array([])


    f = open(rawfile, 'rb')

    if lraw:
        tr = np.fromfile(f, dtype=np.uint64, count=1)
        f.seek(8, 0)
    else:
        f.seek(0, 0)

    if (ts == 0) and (te == -1):
        d = np.frombuffer(f.read(-1), dtype=np.int16)
    else:
        # check time is not negative
        if te-ts < 1:
            raise ValueError('Please recheck ts and te')
        f.seek(ts*2*number_of_channels, 1)
        block_size = (te - ts)*2*number_of_channels
        d = np.frombuffer(f.read(block_size), dtype=np.int16)

    f.close()

    d_bgc = np.reshape(d, [number_of_channels,
                       np.int64(d.shape[0]/number_of_channels)], order='F')

    if lraw:
        d_bgc = ntkc.channel_map_data(d_bgc, number_of_channels,
                                      hstype, nprobes)
        d_bgc = np.int16(d_bgc * gain)

    if lraw:
        return tr, d_bgc
    else:
        return d_bgc


# Load Ecube Digital data
def load_digital_binary(name, t_only=0):

    '''
    load ecube digital data
    load_digital_binary(name)
    returns first timestamp and data
    data has to be 1 channel
    t_only  : if t_only=1, just return tr, timestamp
              (Default 0, returns timestamp and data)
    tdig, ddig =load_digital_binary(digitalrawfile)
    '''

    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    if t_only:
        f.close()
        return tr
    dr = np.fromfile(f, dtype=np.int64,  count=-1)
    f.close()
    return tr, dr
