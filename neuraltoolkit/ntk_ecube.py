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
import time


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
                               lraw=1, ts=0, te=-1,
                               probenum=0, probechans=64):

    '''
    load ecube data and return channel mapped data for single probe
    name - name of file
    number_of_channels - number of channels
    channel_number - channel data to return
    hstype : Headstage type list
             for example ['APT_PCB', 'APT_PCB'] nprobes=2

             Headstage types are
             'hs64'
             'eibless-hs64_port32', 'eibless-hs64_port64',
             'intan32', 'intan16test2',

             'Si_64_KS_chmap',
             'Si_64_KT_T1_K2_chmap'
             Si_64_KS_chmap includes  8-K2., 5-KS., 1A-K2. probe
             Si_64_KT_T1_K2_chmap includes 5-KT. and  5-K2. probe

             'UCLA_Si1'
             'PCB_tetrode', 'EAB50chmap_00',

             'APT_PCB'
             and 'linear'

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
    probename = ["first", "second", "third", "fourth", "fifth",
                 "sixth", "seventh", "eight", "ninth", "tenth"]
    # d_bgc = np.array([])

    if ((number_of_channels/probechans) != nprobes):
        raise ValueError("number of channels/probechans != nprobes")

    # check probe number is in range 0 9
    # max probe in a recording is limitted to 10
    if ((probenum < 0) or (probenum > 9)):
        raise ValueError("Please check probenum {}".format(probenum))

    # Print probe number of probenum is 0 to avoid confusion
    print("Loading data from {} probe\n".format(probename[probenum]))

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
        # convert and -9 to 1 and other values to 0,
        # occurs in Digital_64 files.
        unique = np.unique(dr)
        if -9 in unique:
            print("File {} contains {} values"
                  .format(name, len(unique)))
            print("Converting -9 to 1 and other values to 0: {}"
                  .format(unique))
            dr_corrected = np.zeros_like(dr)
            dr_corrected[np.where(dr == -9)] = 1
            dr = dr_corrected

    # return tr, dr
    return tr, np.int8(dr)


def load_digital_binary_allchannels(name, t_only=0, channel=-1):

    '''
    load ecube digital data all channel
    load_digital_binary(name, t_only=0, channel=-1)
    returns first timestamp and data
    t_only  : if t_only=1, just return tr, timestamp
              (Default 0, returns timestamp and data)
    channel : -1, return all 64 channel, else enter channel number to return
    tdig, ddig =load_digital_binary(digitalrawfile, t_only=0, channel=-1)
    '''
    with open(name, 'rb') as f:
        tr = np.fromfile(f, dtype=np.uint64, count=1)
        if t_only:
            return tr
        # dr = np.fromfile(f, dtype=np.uint64,  count=-1)
        dr_bytes = np.fromfile(f, dtype='<u1')
        dr_bits = np.unpackbits(dr_bytes, bitorder='little')
        dr = dr_bits.reshape((-1, 64))
        dr = dr.T

        if channel == -1:
            return tr, dr
        elif channel > -1:
            return tr, dr[channel, :]


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
    # print("dt t ", t.dtype, flush=True)
    # print("dd d ", d.dtype, flush=True)
    # print("sh d ", d.shape, flush=True)
    # print("ltype ", ltype, flush=True)
    if ltype == 1:
        d = d.transpose()
        # print("dd d ", d.dtype, flush=True)
        # print("sh d ", d.shape, flush=True)
        d = d.flatten()
    # print("dd d ", d.dtype, flush=True)
    # print("sh d ", d.shape, flush=True)
    with open(filename, "wb") as binary_file:
        # Write time
        binary_file.write(struct.pack('Q', t))

        # Write data rawdata/digital
        if ltype == 1:
            # binary_file.write(struct.pack('h'*d.size,
            #                   *d.transpose().flatten()))
            for ds in d:
                binary_file.write(struct.pack('h',
                                  ds))
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


def find_samples_per_chan(rawfile, number_of_channels,
                          lraw=1):

    '''
    Find number of sample points in each channel
    number_of_samples(name, number_of_channels, lraw)
    rawfile - name of file
    number_of_channels - number of channels
    lraw - whether raw file or not  (default : raw lraw=1)
    returns number_of_samples_per_chan
    '''

    # check file
    if not os.path.exists(rawfile):
        raise FileExistsError('Rawfile {} not found'.format(rawfile))

    if lraw:
        number_of_samples_per_chan = \
            (os.path.getsize(rawfile)-8)/2/number_of_channels
    else:
        number_of_samples_per_chan = \
            (os.path.getsize(rawfile))/2/number_of_channels

    return np.int64(number_of_samples_per_chan)


def samples_between_two_binfiles(binfile1, binfile2, number_of_channels,
                                 hstype, nprobes=1, lb1=1, lb2=1,
                                 fs=25000):

    '''
    Calculates number of samples between two ecube raw files
    binfile1 - name of first file
    binfile2 - name of second file
    number_of_channels - number of channels
    hstype : Headstage type, 'hs64'
    nprobes : Number of probes (default 1)
    lb1 : default 1, binfile1 is rawfile, 0 if digital file
    lb2 : default 1, binfile2 is rawfile, 0 if digital file
    returns samples_between

    '''

    # check file
    if not os.path.exists(binfile1):
        raise FileExistsError('binfile {} not found'.format(binfile1))
    if not os.path.exists(binfile2):
        raise FileExistsError('binfile {} not found'.format(binfile2))

    # fact = np.int(fs/1000)

    if lb1:
        t1 = load_raw_binary_gain_chmap(binfile1, number_of_channels,
                                        hstype, nprobes=nprobes,
                                        t_only=1)
    else:
        t1 = load_digital_binary(binfile1, t_only=1, lcheckdigi64=1)

    if lb2:
        t2 = load_raw_binary_gain_chmap(binfile2, number_of_channels,
                                        hstype, nprobes=nprobes,
                                        t_only=1)
    else:
        t2 = load_digital_binary(binfile2, t_only=1, lcheckdigi64=1)
    # samples_between = np.int64((np.int64(t2-t1)/1e6)*fact)
    # print("t1 ", t1, " t2 ", t2)
    # samples_between = np.int64((np.int64(t2-t1)/1e6)*25)
    samples_between = np.float64((np.float64(t2-t1)/1e6)*25)
    # print("samples_between binfiles {} and {} is {}"
    #       .format(binfile1, binfile2, samples_between))
    return samples_between


def delete_probe(name, number_of_channels, hstype, nprobes=1,
                 probenum=0, probechans=64):

    '''
    delete a probes data, replace rawfile
    delete_probe(name, number_of_channels, hstype, nprobes=1,
                 probenum=0, probechans=64)

    name - name of file
    number_of_channels - number of channels
    hstype : Headstage type, 'hs64'
    nprobes : Number of probes (default 1)
    probenum : which probe to delete (starts from zero)
    probechans : number of channels per probe (symmetric)

    '''
    # remove gain after debug
    # gain = np.float64(0.19073486328125)
    # gain = 1

    # Block it until fully debugged
    # raise NotImplementedError('Error: Not implementated, not debugged')

    # check file exists
    if not (os.path.exists(name) and os.path.isfile(name)):
        raise FileNotFoundError("File {} does not exists".format(name))

    # check number_of_channels
    # Ecube recording range is 64-640 currently per probe
    if ((number_of_channels < int(probechans - 1)) |
       (number_of_channels > int(probechans * 10))):
        raise ValueError("number of channels/probechans != nprobes")

    # check hstype
    if isinstance(hstype, str):
        hstype = [hstype]

    assert len(hstype) == nprobes, \
        'length of hstype not same as nprobes'

    # check nprobes
    if ((nprobes < 1) or (nprobes > 10)):
        raise ValueError("Please check nprobes {}".format(nprobes))

    # check probe number is in range 0 9
    # max probe in a recording is limitted to 10
    if ((probenum < 0) or (probenum > 9)):
        raise ValueError("Please check probenum {}".format(probenum))

    # check probechans, currently probechans are only 64
    if not (probechans == 64):
        raise ValueError("Check probechans {}".format(probechans))

    # check number_of_channels factor of nprobes
    if ((number_of_channels/probechans) != nprobes):
        raise ValueError("number of channels/probechans != nprobes")

    # To be clear write probe number in words too
    # Print probe number of probenum is 0 to avoid confusion
    probename = ["first", "second", "third", "fourth", "fifth",
                 "sixth", "seventh", "eight", "ninth", "tenth"]
    # Please make changes in labnotes to reflect deletion of probe
    print("Please make changes in labnotes to reflect deletion of {} probe"
          .format(probename[probenum]))
    print("Deleting data from {} probe\n".format(probename[probenum]))

    # open file
    f = open(name, 'rb')
    tr = np.fromfile(f, dtype=np.uint64, count=1)
    dr = np.fromfile(f, dtype=np.int16,  count=-1)
    f.close()

    length = np.int64(np.size(dr)/number_of_channels)
    drr = np.reshape(dr, [number_of_channels, length], order='F')
    drr = np.int16(drr)
    drr = np.delete(drr, np.arange(((probenum)*probechans),
                                   ((probenum+1)*probechans), 1), axis=0)
    file_string = name.split('_')
    file_string[1] = str(int(number_of_channels - probechans))
    filename = '_'.join(file_string)
    # print("type tr[0] ", type(tr[0]))
    # print("dtype tr[0] ", tr[0].dtype)
    make_binaryfiles_ecubeformat(tr[0], drr,
                                 filename, ltype=1)


def get_digital_event_sample_times(fl_list=None, channel_number=None,
                                   val=0, verbose=0):

    '''
    Get digital event sample times from recording block/session
    can be also used to analyze other digital data too,
    except watchtower

    get_digital_event_sample_times(fl_list, channel_number, val=0, verbose=0)
    fl_list : file list of DigitalPanel_*.bin, same as sorting block
              This works only for new digital binary files
    channel_number : channel number used to record digital event data,
                     (starts from zero)
    val : value to check, in case if digital event is 0 (default)
    verbose: to print logs, default (off)

    returns:
    digital_event_sample_times : sample number when digital event data was 0
    t_estart : Ecube time for first digital file
    nsamples : total number of samples in all DigitalPanel_*.bin files
               in fl_list

    For example:
    import glob
    import numpy as np
    import neuraltoolkit as ntk

    fn_dir = '/home/kbn/ABC1234/ABC1234_2021/'
    fl_list = np.sort(glob.glob(fn_dir + 'DigitalPanel_*.bin'))
    # for second block of sorting (12 hour blocks, each)
    fl_list = fl_list[144:288]
    channel_number = 2
    value = 0
    verbose = 0

    digital_event_sample_times, t_estart, nsamples = \
    ntk.get_digital_event_sample_times(fl_list, channel_number,
                                       val=value, verbose=verbose)
    '''

    # Do checks before proceeding
    if (len(fl_list) == 0):
        raise ValueError('cont_thresh_list list is empty')
    if channel_number is None:
        raise ValueError('channel_number not specified')

    # fl_list = np.sort(glob.glob(fn_dir + 'DigitalPanel_*.bin'))
    print("Channel number starts from zero")
    print("Total number of files is ", len(fl_list))

    # array to store all digital_event data
    digital_event_sample_times = np.array([], dtype=np.int64)
    if verbose:
        tic = time.time()
    nsamples = np.int64(0)
    # loop and load all files in recording session
    for indx, fl in enumerate(fl_list):
        t, d = \
            load_digital_binary_allchannels(fl, t_only=0,
                                            channel=channel_number)
        if indx == 0:
            t_estart = t[0]
        if verbose:
            print(indx, " ", fl, " samples ", d.shape)

        nsamples = nsamples + d.shape[0]
        digital_event_data_a_tmp = np.int64(np.where(d == val)[0])
        if not digital_event_data_a_tmp.size == 0:
            digital_event_data_a_tmp = digital_event_data_a_tmp + nsamples
            digital_event_sample_times = \
                np.concatenate((digital_event_sample_times,
                               digital_event_data_a_tmp),
                               axis=0)
            digital_event_data_a_tmp = None
            # print(digital_event_sample_times)

    if verbose:
        toc = time.time()
        print("Time taken {:4f} seconds".format(toc-tic))

    print("digital_event_sample_times ",  digital_event_sample_times,
          " t_estart ",   t_estart,
          "  nsamples ", nsamples)
    return digital_event_sample_times, t_estart, nsamples


def add_missing_files_with_random_noise(HS_file1, HS_file2,
                                        outdir,
                                        number_of_channels,
                                        hstype,
                                        nprobes,
                                        fs=25000,
                                        total_seconds=300,
                                        lraw=1):
    '''
    add_missing_files_with_random_noise
    import numpy as np
    import neuraltoolkit as ntk

    HS_file1 = '6r/CAF75/Headstages_256_Channels_int16_2021-01-27_19-31-33.bin'
    HS_file2 = '6r/CAF75/Headstages_256_Channels_int16_2021-01-27_20-36-33.bin'
    # outdir = '6s/ckbn/ms/'
    outdir = '6r/CAF00075/CAF00075_2021-01-25_16-36-32/missing_files/'
    number_of_channels = 256
    hstype = ['APT_PCB', 'APT_PCB', 'APT_PCB', 'APT_PCB']
    nprobes = 4
    ntk.add_missing_files_with_random_noise(HS_file1, HS_file2,
                                            outdir, number_of_channels,
                                            hstype=hstype, nprobes=nprobes,
                                            fs=25000, total_seconds=300,
                                            lraw=1)
    '''

    from datetime import datetime
    from datetime import timedelta
    # print("not implemented yet")
    # return

    if lraw:
        gain = np.float64(0.19073486328125)
        data_low = np.round(-5 / gain)
        data_high = np.round(5 / gain)
        file_length = 5  # minutes
        data_type = 'int16'
        msfile = open(os.path.join(outdir, "missing_files.txt"), 'a')

        if os.path.samefile(outdir, os.path.split(HS_file1)[0]):
            raise \
                ValueError('HS_file1 paths {} and outdir {} must be different'.
                           format(HS_file1, outdir))
        if os.path.samefile(outdir, os.path.split(HS_file2)[0]):
            raise \
                ValueError('HS_file2 paths {} and outdir {} must be different'.
                           format(HS_file1, outdir))

        print("HS_file1 ", HS_file1, flush=True)
        e_time_string0 = os.path.split(HS_file1)[1].split('int16_')[0]
        print("e_time_string0 ", e_time_string0)
        print("outdir ", outdir)
        e_time_string1 = HS_file1.split('int16_')[1].replace('.bin', '')
        print(e_time_string1)
        HS_file1_date = datetime.strptime(e_time_string1.replace("_", " "),
                                          "%Y-%m-%d %H-%M-%S")

        total_samples_between =\
            samples_between_two_binfiles(HS_file1,
                                         HS_file2,
                                         number_of_channels,
                                         hstype,
                                         nprobes=nprobes,
                                         lb1=1, lb2=1,
                                         fs=fs)
        print("total_samples_between ", total_samples_between, flush=True)

        HS_file1_samples = find_samples_per_chan(HS_file1, number_of_channels,
                                                 lraw=1)
        print("HS_file1_samples ", HS_file1_samples, flush=True)

        HS_file1_ts = load_raw_binary_gain_chmap(HS_file1,
                                                 number_of_channels,
                                                 hstype, nprobes=nprobes,
                                                 t_only=1)
        print("HS_file1_ts ", HS_file1_ts, flush=True)

        HS_file2_ts = load_raw_binary_gain_chmap(HS_file2,
                                                 number_of_channels,
                                                 hstype, nprobes=nprobes,
                                                 t_only=1)
        print("HS_file2_ts ", HS_file2_ts, flush=True)

        number_of_files_missing = \
            np.round((total_samples_between - HS_file1_samples) /
                     (fs * total_seconds))
        print("number_of_files_missing ", number_of_files_missing, flush=True)

        ts_old = 0
        for indx, i in enumerate(range(np.int(number_of_files_missing - 1))):
            if indx == 0:
                tnext = np.uint64(HS_file1_ts + ((HS_file1_samples * 1e6)/25))
            else:
                tnext = np.uint64(ts_old + ((fs * total_seconds * 1e6)/25))
            print("tnext ", tnext, flush=True)
            ts_old = tnext

            d = None
            d = np.random.randint(data_low, data_high,
                                  (number_of_channels,
                                   (fs * total_seconds)),
                                  dtype=data_type)
            print("sh d ", d.shape, flush=True)
            print("dt d ", d.dtype, flush=True)

            next_time = HS_file1_date + timedelta(minutes=int(5 * (indx + 1)))
            fs_next_time = next_time.strftime('%Y-%m-%d_%H-%M-%S')
            filename = e_time_string0 + fs_next_time + '.bin'
            print("filename ", filename, flush=True)
            filename = os.path.join(outdir, filename)

            print("filename ", filename, flush=True)
            # print("ty tnext ", type(tnext), flush=True)
            # print("sh tnext ", tnext.shape, flush=True)
            # print("ty d ", type(d), flush=True)
            # print("sh d ", d.shape, flush=True)
            # print("ty tnext[0] ", type(tnext[0]), flush=True)
            make_binaryfiles_ecubeformat(tnext[0], d, filename, ltype=1)
            msfile.write(filename + "\t" + str(tnext[0]) + "\n")
            # sys.exit()

        last_file_samples = np.round((total_samples_between -
                                      HS_file1_samples -
                                      ((np.int(number_of_files_missing - 1)) *
                                       fs * total_seconds)))
        print("last_file_samples ", last_file_samples, flush=True)
        # ts_last = np.uint64(ts_old +  (last_file_samples * 1e6)/25)
        ts_last = np.uint64(ts_old + ((fs * total_seconds * 1e6)/25))
        print("ts_last ", ts_last[0], flush=True)

        d = np.random.randint(data_low, data_high, (number_of_channels,
                                                    np.int(last_file_samples)),
                              dtype=data_type)

        next_time = HS_file1_date + timedelta(minutes=int((file_length *
                                                           (indx + 1)) +
                                              np.round(last_file_samples /
                                                       (fs * (total_seconds /
                                                              file_length)))))
        fs_next_time = next_time.strftime('%Y-%m-%d_%H-%M-%S')
        filename = e_time_string0 + fs_next_time + '.bin'
        filename = os.path.join(outdir, filename)
        print("filename ", filename)
        make_binaryfiles_ecubeformat(ts_last[0], d, filename, ltype=1)
        msfile.write(filename + "\t" + str(ts_last[0]) + "\n")

        print("HS_file2 ", HS_file2, flush=True)

        msfile.close()


def check_missing_files(file_list=None, total_seconds=300):
    '''
    check there is missing files in the file list
    '''
    # import os
    import sys
    # import glob
    from datetime import datetime
    # from datetime import timedelta

    # max_delta = 30
    # max_min = 5
    HS_file_start = file_list[0]
    HS_file_end = file_list[-1]

    e_time_string_start = HS_file_start.split('int16_')[1].replace('.bin', '')
    print(e_time_string_start)
    HS_file_start_date = \
        datetime.strptime(e_time_string_start.replace("_", " "),
                          "%Y-%m-%d %H-%M-%S")

    e_time_string_end = HS_file_end.split('int16_')[1].replace('.bin', '')
    print(e_time_string_end)
    HS_file_end_date = datetime.strptime(e_time_string_end.replace("_", " "),
                                         "%Y-%m-%d %H-%M-%S")
    print("HS_file_start_date ", HS_file_start_date,
          " HS_file_end_date ", HS_file_end_date)

    # add 1 for last file
    nfiles = np.round((HS_file_end_date -
                       HS_file_start_date).seconds/total_seconds).astype(int) \
        + 1
    print("nfiles ", nfiles, " len(file_list) ", len(file_list))
    if nfiles != len(file_list):
        #  raise ValueError("Check whether these is any missing files")
        print("Check whether there is any missing files, should be ", nfiles,
              " got ", len(file_list))
        print("Check whether there is any missing files, should be ",
              nfiles, " got ",
              len(file_list), file=sys.stderr)
        sys.stderr.write("Check whether there is any missing files, should be "
                         + str(nfiles) + " got " +
                         str(len(file_list)))
