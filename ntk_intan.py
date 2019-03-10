# -*- coding: utf-8 -*-

'''
Script to read intan binary file
Hlab
Author :Kiran Bhaskaran-Nair

List of functions in ntk_intan

Load raw data apply gain and channel mapping
load_intan_raw_gain_chanmap
help(ntk.load_intan_raw_gain_chanmap)


'''


def load_intan_raw_gain_chanmap(
        rawfile, number_of_channels, hstype, nprobes=1):
    import numpy as np
    from neuraltoolkit import ntk_channelmap as ntkc
    from .load_intan_rhd_format_hlab import read_data

    '''
    load intan data and multiply gain and apply channel mapping
    load_intan_raw_gain_chanmap(name, number_of_channels, hstype)
    hstype : 'hs64', 'intan32', 'Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap'
              and linear
    nprobes : Number of probes (default 1)
    returns first timestamp and data
    '''

    # Read intan file
    a = read_data(rawfile)

    # Time and data
    tr = np.array(a['t_amplifier'][0])
    dg = np.array(a['amplifier_data'])

    # Apply channel mapping
    dgc = ntkc.channel_map_data(dg, number_of_channels, hstype, nprobes)

    return tr, dgc
