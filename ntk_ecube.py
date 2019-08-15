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
import glob


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


def map_videoframes_to_syncpulse(camera_sync_pulse_files, fs=25000):
    '''
    Reads a set of Camera Sync Pulse data files and aggregates the sequences of 000's and 111's into a map of
    video frame numbers to the raw neural data file and offset across all files in a recording. The output
    includes an entry per each sequence of 000's and 111's in the Camera Sync Pulse data.

    Basic usage example:
        output_matrix, pulse_ix, files = ntk.map_videoframes_to_syncpulse('EAB40_Dataset/CameraSyncPulse/*.bin')

    Output matrix format:
     [(     -2,    0,       0,  587,    165967321375)  <- First sequence of 587 000's, ecube time 165967321375
      (     -1,    0,     587,  167,    165990801375)  <- First sequence of 167 111's
      (     -2,    0,     754, 1500,    165997481375)  <- Next sequence of 1500 000's
      (     -1,    0,    2254,  166,    166057481375)  <- Next sequence of 166 111's
      (     -2,    0,    2420, 1500,    166064121375)
      (     -1,    0,    3920,  167,    166124121375)
      ...
      (     -2,    0, 6604445, 1500,    168152116319)
      (     -1,    0, 6605945,  167,    168212116319)
      (     -2,    0, 6606112, 1500,    168218796319)
      (      0,    0, 6607612,  833,    168278796319)  <- Sync pulse, video frame num 0
      (     -2,    0, 6608445,  834,    168312116319)  <- Sequence of 834 000's
      (      1,    0, 6609279,  833,    168345476319)  <- Sequence of 833 111's, frame num 1
      (     -2,    0, 6610112,  833,    168378796319)
      ...
      (5207203, 1157, 7146241,  834, 347290571839839)
      (     -2, 1157, 7147075,  833, 347290605199839)
      (5207204, 1157, 7147908,  833, 347290638519839)
      (     -2, 1157, 7148741,  834, 347290671839839)
      (5207205, 1157, 7149575,  833, 347290705199839)  <- Video frame num 5207205
      (     -2, 1157, 7150408,    8, 347290738519839)] <- Last sequence of 000's, of length 8
       -------  ----  -------  ----  ---------------
         (a)     (b)    (c)     (d)       (e)

    (a) frame_id        - starting with the sync pulse this counts each frame from 0 up.
                          (-1) represents a sequence of 111's prior to the sync pulse
                          (-2) represents a sequence of 000's anywhere
    (b) .bin file index - An index to the appropriate raw .bin file, this is an index into the files return value.
    (c) offset          - The offset into the data array for the bin file reference in (b). Note that this is not an
                          index into the file directly, to get the index to the file use: (offset * 2 + 4), int16 values
                          plus the 4-byte eCube timestamp offset.
    (d) sequence_length - the length of the sequence of 000's or 111's.
    (e) ecube_time      - eCube time for the beginning of this sequence, taken based on the offset from the
                          eCube time recorded at the beginning of each SyncPulse file.

    Detailed explanation of a few entries above:
    (-2, 0, 0, 587, 165967321375):
        (-2) tells us that a sequence of 000's was identified; 0 tells us that it is in the first .bin file (files[0]);
        the sequence starts at (0); the sequence is 587 long; and starts at ecube time (165967321375).
    (-1, 0, 587, 167, 165990801375):
        (-1) tells us that a sequence of 111's before recording was identified. 0 tells us that it is in the
        first .bin file (files[0]); (587) tells us that the sequence of 111's starts at offset 587 in the dr data array;
        (167) tells us that the sequence of 111s is 167 long; and starts at ecube time (165990801375).
    (0, 0, 6607612, 833, 168278796319):
        This entry represents the sync pulse. (0) tells us that this is frame number 0; the second (0) tells us
        that this was found in the first .bin file; (6607612) tells us the offset into the dr data matrix where
        this sequence of 111's starts; (833) tells us that the sequence of 111's is 833 long;
        and starts at ecube time (168278796319).
    (5207205, 1157, 7149575, 833, 347290705199839)
        (5207205) represents the frame number, counting frames from the start of recording across all video files;
        (1157) tells us the .bin file index (files[1157]); (7149575) represents the offset in the .bin file where
        the sequence of 111's starts; (833) represents the length of the sequence of 111's, and starts at
        ecube time (347290705199839).

    FAQs:
     - What is a numpy structured array?
            A structured array is like a numpy matrix, but lets you reference elements by name and it allows
            you to use multiple data types within one matrix.

     - How to get array of just eCube times (equivalently for 'frame_ids', 'binfile_ix', 'offsets', 'sequence_length'):
            output_matrix['ecube_time']

     - How to get all rows with a corresponding frame ID from the output_matrix:
            video_frames_only = output_matrix[output_matrix['frame_ids'] >= 0]

    :param camera_sync_pulse_files: A glob path to the files (example: "EAB40Data/EAB40_Camera_Sync_Pulse/*.bin")
                                    Or a list of filename strings
    :param fs: sampling rate, 25000 default

    :return: tuple(output_matrix, pulse_ix, files)
             output_matrix: is described above, it includes the frame IDs, .bin file reference, offset into .bin
                            files, the sequence length, and ecube time.
             pulse_ix:      the index of the pulse sync in output_matrix (this can also be identified by the entry in
                            output_matrix where the frame ID is 0).
             files:         an ordered list of the camera sync pulse files that were read to produce this result,
                            useful to validate that the correct files were read.

    '''
    assert 1000000000 % fs == 0, 'It is assumed that the sampling rate evenly divides a second, if not this code ' \
                                 'might not function precisely as expected.'

    all_signal_widths = []
    all_file_lengths = []
    all_ecube_times = []
    dr = None
    change_points = None
    ecube_interval = 1000000000 // fs

    files = sorted(glob.glob(camera_sync_pulse_files)) \
        if isinstance(camera_sync_pulse_files, str) else sorted(camera_sync_pulse_files)
    assert isinstance(files, (list, tuple)) and len(files) > 0

    # Initializing with a remainder of 1 will ensure 1 is prepended to the first data file, that allows the first
    # block of 000's to be identified correctly by noticing a change from 1 to 0 at the fist index location.
    remainder = np.array([1])

    # Read the Sync Pulse files iteratively so memory isn't overloaded, compute and save points at which values
    # change between 000/111/000, compute the width of those signals, and compute the length of each file.
    for i, df in enumerate(files):
        t, dr = load_digital_binary(df)

        if i == 0:
            assert dr[0] == 0, 'This algorithm expects the first value of the first file to always be 0.'

        all_ecube_times.append(t)
        all_file_lengths.append(dr.shape[0])
        dr = np.insert(dr, 0, remainder)  # insert the remainder from the last file to the beginning of this one

        change_points = np.where(dr[:-1] != dr[1:])[0]
        signal_widths = (change_points[1:] - change_points[:-1])
        remainder = dr[change_points[-1]:]
        all_signal_widths.append(signal_widths)

    # Accounts for the very last sequence which gets missed in the loop above
    all_signal_widths.append(np.expand_dims(dr.shape[0] - 1 - change_points[-1], axis=0))
    sample_count = np.sum([len(x) for x in all_signal_widths])

    # This will be the final output structured array
    structured_array_dtypes = [
        ('frame_ids', np.int32),
        ('binfile_ix', np.uint32),
        ('offsets', np.uint64),
        ('sequence_length', np.uint32),
        ('ecube_time', np.uint64)
    ]
    output_matrix = np.empty((sample_count,), dtype=structured_array_dtypes)

    # Join all signal widths across all files, same for ecube times
    output_matrix['sequence_length'] = np.concatenate(all_signal_widths, axis=0)
    ecube_times = np.concatenate(all_ecube_times, axis=0)

    # Find the beginning of video recording (findPulse)
    signal_widths_111s = output_matrix['sequence_length'][1::2]
    pulse_ix = np.argmax(signal_widths_111s > 2 * signal_widths_111s[0]) * 2 + 1

    # Compute the frame_id for each signal. Frame id's values are:
    #   -2: Represents a sequence of 000's at any time, both pre and post pulse
    #   -1: Represents a sequence of 111's prior to camera recording
    #   0+: Represents the frame number, starting at 0, counting from the first Pulse signal
    output_matrix['frame_ids'][0::2] = -2
    output_matrix['frame_ids'][1:pulse_ix:2] = -1
    output_matrix['frame_ids'][pulse_ix::2] = np.arange(0, signal_widths_111s.shape[0] - (pulse_ix - 1) // 2)

    # The cumulative sums across files and signal widths allows us to compare offsets between files and signal widths
    cumsum_signal_widths = np.cumsum(output_matrix['sequence_length'])
    cumsum_signal_widths = np.insert(cumsum_signal_widths, 0, 0)[:-1]
    cumsum_files = np.cumsum([np.sum(x) for x in all_file_lengths])
    cumsum_files = np.insert(cumsum_files, 0, 0)[:-1]

    output_matrix['binfile_ix'] = np.searchsorted(cumsum_files, cumsum_signal_widths, side='right') - 1
    output_matrix['offsets'] = cumsum_signal_widths - cumsum_files[output_matrix['binfile_ix']]
    output_matrix['ecube_time'] = ecube_times[output_matrix['binfile_ix']] + ecube_interval * output_matrix['offsets']

    # result = np.column_stack((frame_ids, binfile_ix, offsets, signal_widths, ecube_timestamps))

    return output_matrix, pulse_ix, files
