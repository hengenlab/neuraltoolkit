#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to read intan binary file

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

List of functions/class in ntk_channelmap
load_intan_raw_gain_chanmap(
    rawfile, number_of_channels, hstype, nprobes=1):
'''


import numpy as np


def load_intan_raw_gain_chanmap(
        rawfile, number_of_channels, hstype, nprobes=1, ldin=0):

    '''
    load intan data and multiply gain and apply channel mapping
    load_intan_raw_gain_chanmap(name, number_of_channels, hstype)
    hstype : 'hs64', 'intan32', 'Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap'
              and 'linear'
    nprobes : Number of probes (default 1)
    ldin : Flag for getting digital input (default 0)
    returns first timestamp and data
    '''

    from neuraltoolkit import ntk_channelmap as ntkc
    from .load_intan_rhd_format_hlab import read_data

    if isinstance(hstype, str):
        hstype = [hstype]

    assert len(hstype) == nprobes, \
        'length of hstype not same as nprobes'

    # Read intan file
    a = read_data(rawfile)

    # Time and data
    tr = np.array(a['t_amplifier'][0])
    dg = np.array(a['amplifier_data'])

    # Apply channel mapping
    dgc = ntkc.channel_map_data(dg, number_of_channels, hstype, nprobes)

    if ldin == 1:
        din = np.array(a['board_dig_in_data'])
        return np.int64(tr), np.int16(dgc), din
    else:
        return np.int64(tr), np.int16(dgc)


# Load Intan aux data from ntksorting
def load_aux_binary_data(name, number_of_channels, factor=3.74e-5):

    '''
    load intan aux data preprocessed
    load_aux_binary_data(name, number_of_channels, factor)
    name - name of file
    number_of_channels - number of channels
    factor - 3.74e-5
    returns aux_data
    '''

    f = open(name, 'rb')
    dr = np.fromfile(f, dtype=np.uint16,  count=-1)
    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    f.close()
    print("Factor ", factor)
    drr = drr * factor
    return drr
