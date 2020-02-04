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
light_dark_transition(datadir, l7ampm=0, lplot=0)
'''

try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')
import os


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


def load_raw_gain_chmap_1probe(rawfile, number_of_channels,
                               hstype, nprobes=1,
                               lraw=1, ts=0, te=25000,
                               probenum=0, probechans=64):

    '''
    load ecube data and return channel mapped data for single probe
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
    probenum : which probe to return (starts from zero)
    probechans : number of channels per probe (symmetric)
    returns time, data
    '''

    from neuraltoolkit import ntk_channelmap as ntkc
    # import time

    # tic = time.time()
    gain = np.float64(0.19073486328125)
    # d_bgc = np.array([])

    if ((number_of_channels/probechans) != nprobes):
        raise ValueError("number of channels/probechans != nprobes")

    f = open(rawfile, 'rb')
    # toc = time.time()
    # print('ntk opening file took {} seconds'.format(toc - tic))
    # tic = time.time()

    if lraw:
        tr = np.fromfile(f, dtype=np.uint64, count=1)
        f.seek(8, 0)
    else:
        f.seek(0, 0)

    if (ts == 0) and (te == -1):
        d_bgc = np.frombuffer(f.read(-1), dtype=np.int16)
    else:
        # check time is not negative
        if te-ts < 1:
            raise ValueError('Please recheck ts and te')
        f.seek(ts*2*number_of_channels, 1)
        block_size = (te - ts)*2*number_of_channels
        d_bgc = np.frombuffer(f.read(block_size), dtype=np.int16)

    f.close()
    # toc = time.time()
    # print('ntk reading file took {} seconds'.format(toc - tic))
    # tic = time.time()

    d_bgc = np.reshape(d_bgc, [number_of_channels,
                       np.int64(d_bgc.shape[0]/number_of_channels)], order='F')
    # toc = time.time()
    # print('ntk reshaping file took {} seconds'.format(toc - tic))

    if lraw:
        # tic = time.time()
        d_bgc = d_bgc[((probenum)*probechans):
                      ((probenum+1)*probechans), :]
        # print("d_bgc shape ", d_bgc.shape)
        # toc = time.time()
        # print('ntk probe selecting file took {} seconds'.format(toc - tic))
        # tic = time.time()
        # d_bgc = ntkc.channel_map_data(d_bgc, number_of_channels,
        #                               hstype, nprobes)
        d_bgc = ntkc.channel_map_data(d_bgc, probechans,
                                      list([hstype[probenum]]), 1)
        # toc = time.time()
        # print('ntk channel mapping file took {} seconds'.format(toc - tic))

        # tic = time.time()
        d_bgc = np.int16(d_bgc * gain)
        # toc = time.time()
        # print('ntk adding gain to file took {} seconds'.format(toc - tic))

    if lraw:
        return tr, d_bgc
    else:
        return d_bgc


# Load Ecube Digital data
def load_digital_binary(name, t_only=0, lcheckdigi64=1):

    '''
    load ecube digital data
    load_digital_binary(name)
    returns first timestamp and data
    data has to be 1 channel
    t_only  : if t_only=1, just return tr, timestamp
              (Default 0, returns timestamp and data)
    lcheckdigi64 : Default 1, check for digital file with 64 channel
              accidental recordings and correct -11->0 and -9->1
              lcheckdigi64=0, no checks are done read the file and
              return output.
    lcheckdigi64 : Default 1, check for digital file with 64 channel
             (atypical recordings) and correct values of -11 to 0 and -9 to 1
             lcheckdigi64=0, no checks are done, just read the file and
             returns timestamp and data
    tdig, ddig =load_digital_binary(digitalrawfile)
    '''

    with open(name, 'rb') as f:
        tr = np.fromfile(f, dtype=np.uint64, count=1)
        if t_only:
            return tr
        dr = np.fromfile(f, dtype=np.int64,  count=-1)

    if lcheckdigi64:
        # convert and -9 to 1 and other values to 0, occurs in Digital_64 files.
        unique = np.unique(dr)
        if -9 in unique:
            print("File {} contains {} values, converting -9 to 1 and other values to 0: {}"
                  .format(name, len(unique), unique))
            dr_corrected = np.zeros_like(dr)
            dr_corrected[np.where(dr == -9)] = 1
            dr = dr_corrected

    # return tr, dr
    return tr, np.int8(dr)


def light_dark_transition(datadir, l7ampm=0, lplot=0):
    '''
    light_dark_transition
    light_dark_transition(datadir, l7ampm=0, lplot=0)
    datadir - location of digital light data
    l7ampm - just check files around 7:00 am and 7:00 pm
            (default 0, check all files), 1 check files only around 7:00 am/pm
    lplot - plot light to dark transition points (default 0, noplot), 1 to plot
    returns
    transition_list - list contains
                      [filename, index of light-dark transition, time]
    transition_list = light_dark_transition('/media/bs003r/D1/d1_c1/',
                                            l7ampm=0, lplot=0)

    '''

    import matplotlib.pyplot as plt
    import os

    # add flag to check just at time 07:30 and 19:30
    if l7ampm:
        sub_string = ["_19-", "_07-"]
    else:
        sub_string = ['Digital', 'Digital']

    save_list = []

    for f in np.sort(os.listdir(datadir)):
        if sub_string[0] in f or sub_string[1] in f:
            # print(f)

            # load data
            t, d1 = load_digital_binary(datadir+f)

            # find unique values
            unique_val = np.unique(d1)

            # transition files has 0 and 1 values
            if all(x in unique_val for x in [0, 1]):
                # print(f, " ", np.unique(d1), " ", d1.shape)

                # Find diff
                d_diff = np.diff(d1)

                # -1 light of
                if -1 in d_diff:
                    transition = np.where(np.diff(d1) == -1)[0][0]
                # 1 for light on
                elif 1 in d_diff:
                    transition = np.where(np.diff(d1) == 1)[0][0]
                # raise error
                else:
                    raise ValueError('Unkown Value, ntk_light_pulse')

                # print("transition)

                # add filename, transition index and time of the file
                print("Filename", f, " index ",  transition, " time ", t[0])
                save_list.append([f, transition, t[0]])

                # plot
                if lplot:
                    plt.figure()
                    plt.plot(d1[transition-1:transition+2], 'ro')

    return save_list


def make_binaryfiles_ecubeformat(t, d, filename, ltype=2):

    '''
    make_binaryfiles_ecubeformat
    make_binaryfiles_ecubeformat(t, d, filename, ltype=2)
    t -  time in nano seconds to be written to binary file
    d - numpy array to be converted to binary file
    filename - file name to write with path
    ltype - digital files (default 2 digital files)
            for rawdata files 1
    returns

    '''
    import struct
    import os

    # check ltype is 1 or 2
    assert 0 < ltype < 3, 'Unknown ltype'

    # Exit if file exist already
    if os.path.exists(filename):
        raise FileExistsError('filename already exists')

    if ltype == 2:
        assert d.shape[0] == 1, 'Only supports 1-d array'
        assert len(np.unique(d)) <= 2, 'Supports only 1 and 0'

    # Open file to write
    with open(filename, "wb") as binary_file:
        # Write time
        binary_file.write(struct.pack('L', t))

        # Write data rawdata/digital
        if ltype == 1:
            binary_file.write(struct.pack('h'*d.size,
                              *d.transpose().flatten()))
        elif ltype == 2:
            binary_file.write(struct.pack('q'*d.size,  *d.flatten()))
        else:
            raise ValueError('Unkown value')


def visual_grating_transition(datadir):
    '''
    visual_grating_transition
    visual_grating_transition(datadir)
    datadir - location of digital gratings data
    returns
    transition_list - list contains
                      [filename, index of grating transitions, time]
    transition_list = visual_grating_transition('/media/bs003r/D1/d1_vg1/')

    '''

    import os

    # check only Digital files
    sub_string = ['Digital', 'Digital']

    save_list = []

    for f in np.sort(os.listdir(datadir)):
        if sub_string[0] in f or sub_string[1] in f:
            # print(f)

            # load data
            t, d1 = load_digital_binary(datadir+f)

            # find unique values
            unique_val = np.unique(d1)

            # transition files has 0 and 1 values
            if all(x in unique_val for x in [0, 1]):
                # print(f, " ", np.unique(d1), " ", d1.shape)

                # Find diff
                d_diff = np.diff(d1)

                # 1 for gratings on
                if 1 in d_diff:
                    transition = np.where(np.diff(d1) == 1)[0]

                # add filename, transition index and time of the file
                print("Filename", f, " index ",  transition, " time ", t[0])
                save_list.append([f, transition, t[0]])

    return save_list


# Convert ecube_raw_to_preprocessed all channels or just a tetrode
# To add
#   1.split by probes
#   2.Same name as ntksorting
def ecube_raw_to_preprocessed(rawfile, outdir, number_of_channels,
                              hstype, nprobes=1,
                              ts=0, te=25000,
                              tetrode_channels=None):

    '''
    Convert ecube_raw_to_preprocessed all channels or just a tetrode
    ecube_raw_to_preprocessed(rawfile, outdir, number_of_channels,
                              hstype, nprobes=1,
                              ts=0, te=25000,
                              tetrode_channels=None)
    rawfile - name of rawfile with path, '/home/ckbn/Headstage.bin'
    outdir - directory to save preprocessed file, '/home/ckbn/output/'
    number_of_channels - number of channels in rawfile
    hstype : Headstage type, 'hs64' (see manual for full list)
    nprobes : Number of probes (default 1)
    ts : sample start (not seconds but samples),
           default : 0 ( begining of file)
    ts : sample end (not seconds but samples),
           default : -1 ( full data)
    tetrode_channels : default (None, all channels)
                     to select tetrodes give channel numbers as a list
                     Example: tetrode_channels=[4, 5, 6, 7] for second tetrode
    returns
    '''

    from neuraltoolkit import ntk_channelmap as ntkc
    import struct

    gain = np.float64(0.19073486328125)
    d_bgc = np.array([])

    # Check file exists
    if not (os.path.isfile(rawfile)):
        raise FileExistsError('Raw file not found')

    # Check outdir exists
    print("os.path.isdir(outdir) ", os.path.isdir(outdir))
    if not (os.path.isdir(outdir)):
        raise FileExistsError('Output directory does not exists')

    # Get pfilename
    pbasename = str('P_') + os.path.splitext(os.path.basename(rawfile))[0] \
        + str('.bin')
    print("pbasename ", pbasename)
    pfilename = os.path.join(outdir, pbasename)
    print("pfilename ", pfilename)

    # Read binary file and order based on channelmap
    f = open(rawfile, 'rb')

    tr = np.fromfile(f, dtype=np.uint64, count=1)
    print("Time ", tr)
    f.seek(8, 0)

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

    d_bgc = ntkc.channel_map_data(d_bgc, number_of_channels,
                                  hstype, nprobes)
    d_bgc = np.int16(d_bgc * gain)

    if tetrode_channels:
        print("tetrode_channels ", tetrode_channels)
        d_bgc = d_bgc[tetrode_channels, :]
    else:
        print("tetrode_channels is empty saving all tetrodes")

    # write preprocessed file
    with open(pfilename, "wb") as binary_file:
        # Write data rawdata/digital
        binary_file.write(struct.pack('h'*d_bgc.size,
                          *d_bgc.transpose().flatten()))
