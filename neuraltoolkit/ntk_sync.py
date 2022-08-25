# flake8: noqa
""" Tools for syncing between video, neural, and sleep states. """
import glob
import itertools
import os
import re
import numpy as np
import neuraltoolkit as ntk
import argparse
import csv
import configparser
import h5py
import distutils.util
import importlib
import urllib.parse
import functools
import multiprocessing


# optional import for S3 file support, for S3 file support: pip install smart_open awswrangler
try:
    import awswrangler as wr
    import smart_open
    import boto3

    assert int(smart_open.__version__[0]) >= 5, "Smart open 5.1.0+ is required."
    endpoint_url = os.environ.get('ENDPOINT_URL', 'https://s3-west.nrp-nautilus.io')
    wr.config.s3_endpoint_url = endpoint_url
    transport_params = {'client': boto3.Session().client('s3', endpoint_url=endpoint_url)}
    smart_open.open = functools.partial(smart_open.open, transport_params=transport_params)
except ImportError as ie:
    pass


def map_videoframes_to_syncpulse(syncpulse_files: (str, list), fs: int = 25000):
    '''
    Reads a set of Camera Sync Pulse data files and aggregates the sequences of 000's and 111's into a map of
    video frame numbers to the raw neural data file and offset across all files in a recording. The output
    includes an entry per each sequence of 000's and 111's in the Camera Sync Pulse data.

    EXAMPLE USAGE:
        output_matrix, pulse_ix, files = ntk.map_videoframes_to_syncpulse('EAB40_Dataset/CameraSyncPulse/*.bin')

    Output matrix format:
     [(     -2,    0,       0,  587,    165967321375)  <- First sequence of 587 000's, ecube time 165967321375
      (     -1,    0,     587,  167,    165990801375)  <- First sequence of 167 111's
      (     -2,    0,     754, 1500,    165997481375)  <- Next sequence of 1500 000's
      (     -1,    0,    2254,  166,    166057481375)  <- Next sequence of 166 111's
      (     -2,    0,    2420, 1500,    166064121375)
      (     -1,    0,    3920,  167,    166124121375)
      ...
      (     -2,    0, 6604445, 1500,    430145121375)
      (     -1,    0, 6605945,  167,    430205121375)
      (     -2,    0, 6606112, 1500,    430211801375)
      (      0,    0, 6607612,  833,    430271801375)  <- Sync pulse, video frame num 0
      (     -2,    0, 6608445,  834,    430305121375)  <- Sequence of 834 000's
      (      1,    0, 6609279,  833,    430338481375)  <- Sequence of 833 111's, frame num 1
      (     -2,    0, 6610112,  833,    430371801375)
      ...
      (5207203, 1157, 7146241,  834, 347573777601375)
      (     -2, 1157, 7147075,  833, 347573810961375)
      (5207204, 1157, 7147908,  833, 347573844281375)
      (     -2, 1157, 7148741,  834, 347573877601375)
      (5207205, 1157, 7149575,  833, 347573910961375)  <- Video frame num 5207205
      (     -2, 1157, 7150408,    8, 347573944281375)] <- Last sequence of 000's, of length 8
       -------  ----  -------  ----  ---------------
         (a)     (b)    (c)     (d)       (e)

    (a) frame_id        - starting with the sync pulse this counts each frame from 0 up.
                          (-1) represents a sequence of 111's prior to the sync pulse
                          (-2) represents a sequence of 000's anywhere
    (b) .bin file index - An index to the appropriate raw .bin file, this is an index into the files return value.
    (c) offset          - The offset into the data array for the bin file reference in (b). Note that this is not an
                          index into the file directly, to get the index to the file use: (offset * n_channels * 2 + 8),
                          int16 values, number of channels, plus the 4-byte eCube timestamp offset.
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

    eCube timestamp issue note:
        It has been determined that there is an error in the eCube timestamp in the Camera Pulse files
        following the first, this can be confirmed by computing the expected eCube timestamp for the first 3
        SyncPulse files and comparing them to what is actually in the file. The 2nd SyncPulse file will
        be off by 0.145s, but the third will be correct. We can see in the output of this function that there
        is no loss of data because the sync pulse widths are maintained perfectly across files. The issue is
        that all eCube times after the first file will be offset 0.145s forward from where they should be.
        To side-step this issue this code computes eCube time ONLY from the first files eCube time.
        Notably this issue has not been noticed in the raw neural files.

    :param syncpulse_files: sync pulse filenames, may be a list or single string of filenames or globs.
    :param fs: sampling rate, 25000 default
    :return: tuple(output_matrix, pulse_ix, files)
             output_matrix:   is described above, it includes the frame IDs, .bin file reference, offset into .bin
                              files, the sequence length, and ecube time.
             pulse_ix:        the index of the pulse sync in output_matrix (this can also be identified by the entry in
                              output_matrix where the frame ID is 0).
             syncpulse_files: an ordered list of the camera sync pulse files that were read to produce this result,
                              useful to validate that the correct files were read.

    '''
    assert 1000000000 % fs == 0, 'It is assumed that the sampling rate evenly divides a second, if not this code ' \
                                 'might not function precisely as expected.'

    # Resolve all inputs to lists of files
    syncpulse_files = _resolve_files(syncpulse_files)

    all_signal_widths = []
    all_file_lengths = []
    ecube_start_time = None
    dr = None
    change_points = None
    ecube_interval = 1000000000 // fs

    # Initializing with a remainder of 1 will ensure 1 is prepended to the first data file, that allows the first
    # block of 000's to be identified correctly by noticing a change from 1 to 0 at the fist index location.
    remainder = np.array([1])

    # Read the Sync Pulse files iteratively so memory isn't overloaded, compute and save points at which values
    # change between 000/111/000, compute the width of those signals, and compute the length of each file.
    for i, df in enumerate(syncpulse_files):
        t, dr = ntk.load_digital_binary_allchannels(df)
        dr = check_format_digital(dr)
        if i == 0:
            assert dr[0] == 0, 'This algorithm expects the first value of the first digital file to always be 0.'
            ecube_start_time = t

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

    # Join all signal widths across all files
    output_matrix['sequence_length'] = np.concatenate(all_signal_widths, axis=0)

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
    cumsum_files = np.cumsum(all_file_lengths)
    cumsum_files = np.insert(cumsum_files, 0, 0)[:-1]

    output_matrix['binfile_ix'] = np.searchsorted(cumsum_files, cumsum_signal_widths, side='right') - 1
    output_matrix['offsets'] = cumsum_signal_widths - cumsum_files[output_matrix['binfile_ix']]
    # See function docs to understand why ecube_time is only computed relative to the first file and not from all files.
    output_matrix['ecube_time'] = ecube_start_time + cumsum_signal_widths * ecube_interval

    return output_matrix, pulse_ix, syncpulse_files


def map_video_to_neural_data(syncpulse_files: (list, tuple, str),
                             video_files: (list, tuple, str),
                             neural_files: (list, tuple, str),
                             sleepstate_files: (list, tuple, str) = (),
                             dlclabel_files: (list, tuple, str) = (),
                             fs: int = 25000, n_channels: int = None,
                             neural_bin_files_per_sleepstate_file: int = 12,
                             manual_video_neural_offset_sec: float = 0.0,
                             ignore_dlc_label_mismatch: bool = False,
                             initial_neural_file_for_sleepstate_mapping: int = 0,
                             ignore_ecube_deviation_sec: float = 0.00012):
    """
    Maps video to neural data and optionally maps sleeps state and DLC labels.

    EXAMPLE USAGE:
        output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files, dlclabel_files = \
            ntk.map_video_to_neural_data(
                syncpulse_files='EAB40Data/EAB40_Camera_Sync_Pulse/*.bin'
                video_files=['EAB40Data/EAB40_Video/3_29-4_02/*.mp4',
                             'EAB40Data/EAB40_Video/4_02-4_05/*.mp4'],
                neural_files='EAB40Data/EAB40_Neural_Data/3_29-4_02/*.bin',
                dlclabel_files='EAB40Data/EAB40_DLC_Labels/*.h5'
                sleepstate_files='EAB40Data/EAB40_Sleep_States/*.npy'
            )

    * Note that neural_files do not need to be available, they can be substituted with a single CSV file listing
      all neural files, eCube timestamps, and file sizes, instead of requiring all
      raw neural data files (which are large), see the neural_files function parameter docs below for details.

    output_matrix format:

    [(             0,        0,   0,     0,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),  (  i)
     (             0,        1,   0,     1,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (             0,        2,   0,     2,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     ...
     (             0,   555118,  10, 15127,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (             0,   555119,  10, 15128,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (  430271801375,   555120,  10, 15129,   0,  427068,  2, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),  ( ii)
     (  430338481375,   555121,  10, 15130,   0,  428735,  2, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     ...`
     (  530508401375,   556624,  10, 16633,   0, 2932983,  2, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (  530575041375,   556625,  10, 16634,   0, 2934649,  2, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (  530641721375,   556626,  11,     0,   0, 2936316,  2, [242.85,  79.17,   0.95, 239.24,  77.09,   0.  ]),  (iii)
     (  530708401375,   556627,  11,     1,   0, 2937983,  2, [242.65,  78.98,   0.96, 239.19,  77.01,   0.  ]),
     (  530775041375,   556628,  11,     2,   0, 2939649,  2, [242.43,  78.82,   0.96, 239.16,  76.85,   0.  ]),
     ...
     (87119094361375,  1855466,  35,  2860, 288, 7497370, -1, [290.42, 165.84,   1.  , 243.44, 186.94,   1.  ]),
     (87119161041375,  1855467,  35,  2861, 288, 7499037, -1, [290.58, 165.39,   1.  , 243.37, 186.47,   1.  ]),  ( iv)
     (87119227721375,  1855468,  35,  2862,  -1,      -1, -1, [290.58, 165.75,   1.  , 243.37, 186.56,   1.  ]),  (  v)
     (87119294361375,  1855469,  35,  2863,  -1,      -1, -1, [290.7 , 165.97,   1.  , 243.42, 186.93,   1.  ]),  ( vi)
     ...
     (             0, 10179126, 189, 53997,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ]),
     (             0, 10179127, 189, 53998,  -1,      -1, -1, [ -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ,  -1.  ])]  (vii)
      --------------  --------  ---  -----  ---  ------- ---- ------------------------------------------------
           (a)           (b)    (c)   (d)   (e)    (f)   (g)                     (h)

    (  i) First video frame has no sleep state, no DLC data, and no neural data, also no ecube_time
    ( ii) Video frame 555,120, first frame with neural & sleep
    (iii) First frame with DLC (Deep lab cut) labels (check for a positive value in column h), also the first video
          frame of video_files[11].
    ( iv) Last video frame with neural & sleep data
    (  v) Video frames beyond neural & sleep data (-1)
    ( vi) Last video frame where DLC labels exist.
    (vii) Last video frame, note that there is no ecube_time (neural recording stopped)

    (a) ecube_time            - The exact ecube time when this frame began the write process, 0 indicates no neural data exists
    (b) video_frame_global_ix - The video frame index (0 based) counting across all video files (a global frame counter)
    (c) video_filename_ix     - Index to the video filename returned in video_files
    (d) video_frame_offset    - The video frame index (0 based) counting from the current file
    (e) neural_filename_ix    - Index to the raw neural filename returned in neural_files
    (f) neural_offset         - Sample offset in the neural file reference in neural_filename_ix
                                Note that this is not an index into the file directly, to get the index to the file use:
                                (neural_offset * n_channels * 2 + 8), int16 values plus the 8-byte eCube timestamp offset
    (g) sleep_state           - Sleep state (1=wake, 2=NREM, 3=REM)
    (h) dlc_label[6]          - DLC label which is an array of 6 Deep Lab Cut label values reduced to float32 precision.

    Examples of common filters:

        # Get all frames with valid neural data (non negative):
        output_matrix[np.positive(output_matrix['neural_filename_ix'])]

        # Get all frames with valid DLC data (non negative, looking at just the first DLC value):
        output_matrix[np.positive(output_matrix['dlc_label'][:, 0])]

        # Get all video frames from file index #11:
        output_matrix[output_matrix['video_filename_ix'] == 11]

        # Get the filename of video file index #11, note that video_files is one of the function return values
        video_files[11]

    Issues:
      - This function should fail on an assert if the eCube timestamps on the neural data files is
        inconsistent. If it does you need to fix the eCube timestamp in the neural data file, this code
        requires the eCube timestamps to be consistent and contiguous. This is known to happen in practice.
      - This function depends on the neural data file size to be correct (it uses file size to compute how
        many data samples are in a neural data file). If a neural data file is missing or corrupted you
        can work around the problem by creating a file with the appropriate size and manually adding a
        timestamp using numpy. For example, in the EAB40 dataset the first data file is missing and
        is an example where the file Headstages_256_Channels_int16_2019-03-29_10-28-28.bin was created
        manually with all 0's and an appropriate eCube timestamp, look to this case if you need an example.

    :param syncpulse_files: sync pulse filenames, may be a list or single string of filenames or globs.
    :param video_files: video filenames, may be a list or single string of filenames or globs.
    :param neural_files: raw neural files, may be a list or single string of filenames or globs, or a
           bill of materials CSV file that describes the set of neural datafiles
           in the form: eCube_time, file_size, neural_filename.
    :param sleepstate_files: Optional sleep state numpy files, may be a list or single string of filenames or globs.
    :param dlclabel_files: Optional DLC label numpy files, may be a list or single string of filenames or globs.
    :param fs: sample rate, default = 25000
    :param n_channels: None == autodetect. Number of channels in data files, if unspecified (None) the
           number of channels will be inferred from the filename of the first neural .bin file.
    :param neural_bin_files_per_sleepstate_file: Number of neural binary files that are covered by each sleep state
           numpy file. For example in EAB40 there are 12 neural files per sleep state file, therefore the
           parameter is 12; in SCF05 there is only one sleep state file for ALL neural data files, in this case the
           parameter is -1 for ALL. The default is 12, which is 1hr of sleep state data per numpy file.
    :param manual_video_neural_offset_sec: in some cases the sync pulse doesn't properly identify the start of
           video recording. You can manually sync video to the neural data using file timestamps (see ntk documentation
           page for a guide on doing this). This value overrides the SyncPulse time relative to the start of neural
           data. Enter the video offset from neural data in seconds. In most cases video is started after neural
           data so this value will be positive, if video started before neural this value will be negative.
    :param ignore_dlc_label_mismatch: ignore cases when the DLC label count doesn't match the video frame count,
           this issue has been observed in some cases with DLC labels that are split. Using this option will
           leave remainder frames unlabeled and will constitute a small mismatch between DLC labels and frames.
    :param initial_neural_file_for_sleepstate_mapping: If sleep state data doesn't map to the first neural file
           present in the dataset, set this parameter to the number of neural files to skip before mapping to the
           sleep state data. For example, EAB50 maps sleep state to location 228 (0 indexed), which is file:
           Headstages_512_Channels_int16_2019-06-21_12-05-11.bin
    :param ignore_ecube_deviation_sec: A deviation in ecube times indicates that the number of samples found
           in the .bin file differs from the ecube timestamp found in the beginning of the file compared to the
           timestamp of the next file. Ideally the number of samples will exactly match the ecube times reported
           across files. Small deviations of the default 0.00012 sec deviation equate to a maximum of 3 samples
           being dropped. Any deviation above this value will report a detail error and fail the process.
           Increase this value to ignore larger deviations, being aware that the deviation may
           represent a problem in the data and in mapping the neural data to video, labels, etc.
    :return: output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files,
             camera_pulse_output_matrix, pulse_ix
    """
    # Resolve all inputs to lists of files
    syncpulse_files = _resolve_files(syncpulse_files)
    video_files = _resolve_files(video_files)
    neural_files = _resolve_files(neural_files)
    sleepstate_files = _resolve_files(sleepstate_files)
    dlclabel_files = _resolve_files(dlclabel_files)

    # Resolve neural files to a bill-of-materials list containing the ecube timestamps
    neural_files_bom = _resolve_neural_files_bom(neural_files)

    # Resolve n_channels from the first neural filename if n_channels is None
    # Example expected filename: Headstages_64_Channels_int16_2018-12-05_21-42-37.bin
    n_channels = n_channels if isinstance(n_channels, int) else int(re.findall(r'_(\d*)_Channels', neural_files_bom[0][2])[0])

    # Validation
    assert len(syncpulse_files) > 0, 'Found no syncpulse_files.'
    assert len(neural_files) > 0, 'Found no neural_files.'
    assert neural_bin_files_per_sleepstate_file == -1 or neural_bin_files_per_sleepstate_file >= 1, \
        'neural_bin_files_per_sleepstate_file must be -1 or a value of 1 or more, value found: {}'\
        .format(neural_bin_files_per_sleepstate_file)
    if len(video_files) == 0:
        print(f'***\n*** Found no video_files {video_files}, using Digital Channel data to choose video frames for syncing.\n***')

    # output_matrix data types definition
    structured_array_dtypes = [
        ('ecube_time', np.uint64),
        ('video_frame_global_ix', np.uint32),
        ('video_filename_ix', np.int32),
        ('video_frame_offset', np.uint32),
        ('neural_filename_ix', np.int32),
        ('neural_offset', np.int64),
        ('sleep_state', np.int8),
        ('dlc_label', (np.float32, 6)),
    ]
    ecube_interval = 1000000000 // fs

    # Extract just the eCube timestamp from each raw neural recording file
    neural_ecube_timestamps = [int(row[0]) for row in neural_files_bom]
    neural_sample_counts = [(int(row[1]) - 8) // (2 * n_channels) for row in neural_files_bom]
    neural_files = [row[2] for row in neural_files_bom]

    neural_ecube_timestamps = np.array(neural_ecube_timestamps)
    neural_sample_counts = np.array(neural_sample_counts)
    neural_ecube_last_timestamp = \
        neural_ecube_timestamps[-1] + (ecube_interval * neural_sample_counts[-1]).astype(np.uint64)

    # Validate eCube timestamps are correct, issue a warning if they are significantly outside expectation
    # This is necessary because eCube timestamps have been observed to be erroneous.
    validate_neural_ecube_timestamps = np.abs(
        (neural_sample_counts[:-1] * ecube_interval) -
        (neural_ecube_timestamps[1:] - neural_ecube_timestamps[:-1]).astype(np.int64)
    )
    acceptable_ecube_error = 1e9 * ignore_ecube_deviation_sec
    if np.any(np.abs(validate_neural_ecube_timestamps) > acceptable_ecube_error):
        for i in np.where(validate_neural_ecube_timestamps > acceptable_ecube_error)[0]:
            print(
                'WARNING: eCube timestamp deviation of {:.4f} seconds for file {}, the eCube timestamp does not match '
                'expectation based on the eCube timestamp of the previous file, eCube time of this file is {:d}, '
                'and the previous eCube timestamp is {:d}. The expected eCube time for this file is {}.'
                .format(validate_neural_ecube_timestamps[i] / 1000000000,
                        neural_files[i + 1],
                        neural_ecube_timestamps[1:][i],
                        neural_ecube_timestamps[i],
                        neural_sample_counts[:-1][i] * ecube_interval + neural_ecube_timestamps[i])
            )
        raise AssertionError('Invalid eCube timestamp detected, this will cause errors in indexing, '
                             'see previous warning messages for details.')

    # Compute a precise eCube time per video frame, or 0 where no SyncPulse data exists
    camera_pulse_output_matrix, pulse_ix, _ = map_videoframes_to_syncpulse(syncpulse_files, fs)

    # Count frames in all video files
    frame_counts = np.array([round(ntk.NTKVideos(vfile, 0).length) for vfile in video_files]) \
        if len(video_files) > 0 else np.expand_dims(np.max(camera_pulse_output_matrix['frame_ids']), axis=0)
    cumsum_frame_counts_partial = np.insert(np.cumsum(frame_counts), 0, 0)[:-1]

    # total_frames taken from video files if they exist, if not it's taken from digital file
    total_frames = np.sum(frame_counts)
    output_matrix = np.empty((total_frames,), dtype=structured_array_dtypes)

    if manual_video_neural_offset_sec != 0.0:
        # Manually change ecube times computed by map_videoframes_to_syncpulse if user specified the offset manually
        neural_ecube_start = neural_ecube_timestamps[0]
        video_ecube_syncpulse = camera_pulse_output_matrix[pulse_ix]['ecube_time']
        neural_video_diff = video_ecube_syncpulse - neural_ecube_start
        expected_neural_video_diff = manual_video_neural_offset_sec * 1e9
        increment_video_diff = expected_neural_video_diff - neural_video_diff
        new_ecube_time = camera_pulse_output_matrix['ecube_time'].astype(np.int64) + int(increment_video_diff)
        new_ecube_time[new_ecube_time < 0] = 0
        camera_pulse_output_matrix['ecube_time'] = new_ecube_time.astype(np.uint64)

    valid_frames = (camera_pulse_output_matrix['frame_ids'] >= 0) & (camera_pulse_output_matrix['frame_ids'] < total_frames)

    frames_from = 0
    frames_to = np.sum(valid_frames)
    output_matrix['ecube_time'] = 0  # frames we don't have data for have 0 ecube_time
    output_matrix['ecube_time'][frames_from:frames_to] = camera_pulse_output_matrix['ecube_time'][valid_frames]

    output_matrix['video_frame_global_ix'] = np.arange(0, total_frames)
    assert np.all(np.diff(cumsum_frame_counts_partial) >= 0), \
        'cumsum_frame_counts_partial are not sorted, this condition should not occur under normal conditions.'
    output_matrix['video_filename_ix'] = \
        np.searchsorted(cumsum_frame_counts_partial, output_matrix['video_frame_global_ix'], side='right') - 1
    output_matrix['video_frame_offset'] = \
        output_matrix['video_frame_global_ix'] - cumsum_frame_counts_partial[output_matrix['video_filename_ix']]

    # Compute the neural file index and offset for that file, noting that some frames will be recorded before neural
    # data recording started, in those cases the neural_filename_ix and neural_offset will be -1
    assert np.all(np.diff(neural_ecube_timestamps) >= 0), \
        'neural_ecube_timestamps are not sorted, this condition should not occur under normal conditions.'
    output_matrix['neural_filename_ix'] = \
        np.searchsorted(neural_ecube_timestamps, output_matrix['ecube_time'], side='left') - 1
    valid_f = np.logical_and(
        output_matrix['neural_filename_ix'] >= 0,
        output_matrix['ecube_time'] < neural_ecube_last_timestamp,
    )

    # Invalidate any locations where neural data isn't available (either before or after video frames)
    output_matrix['neural_offset'][valid_f] = \
        (output_matrix['ecube_time'][valid_f] - neural_ecube_timestamps[output_matrix['neural_filename_ix'][valid_f]]) \
        // ecube_interval
    output_matrix['neural_filename_ix'][~valid_f] = -1
    output_matrix['neural_offset'][~valid_f] = -1

    #
    # Map in sleep states based on eCube time
    #
    if len(sleepstate_files) > 0:
        s = len(neural_ecube_timestamps) if neural_bin_files_per_sleepstate_file == -1 \
            else neural_bin_files_per_sleepstate_file
        neural_ecube_timestamps_per_sleep_state_file = neural_ecube_timestamps[initial_neural_file_for_sleepstate_mapping::s]
        # Compute neural_sample_counts_per_sleep_state_file to compute end eCube times per neural ecube time
        neural_sample_counts_per_sleep_state_file = [
            np.sum(nsc) for nsc in np.split(neural_sample_counts, range(s, len(neural_sample_counts), s))
        ]
        sleep_state_list = [np.load(sfile) for sfile in sleepstate_files]
        assert len(neural_ecube_timestamps_per_sleep_state_file) >= len(sleep_state_list), \
            'There are more sleep state files than neural recording files.'
        sleep_state_ecube_times_list = []
        for i in range(len(sleep_state_list)):
            start_ecube_time = neural_ecube_timestamps_per_sleep_state_file[i]
            stop_ecube_time = neural_ecube_timestamps_per_sleep_state_file[i] + \
                              neural_sample_counts_per_sleep_state_file[i] * ecube_interval
            sleep_state_ecube_times_list.append(np.linspace(
                start=start_ecube_time, stop=stop_ecube_time, num=sleep_state_list[i].shape[0], dtype=np.uint64
            ))
            # Validate that the eCube range used is within +/- 4 seconds of the number of sleep state samples
            ecube_timestamp_range_sec = (stop_ecube_time - start_ecube_time) / 1e+9
            sleep_state_range_sec = sleep_state_list[i].shape[0] * 4
            assert abs(ecube_timestamp_range_sec - sleep_state_range_sec) < 4, \
                'The range of eCube timestamps across {} neural files ({:0.1f} seconds) does not match the time ' \
                'range covered by sleep state file [{}] ({:0.1f} seconds)'.format(
                    s, ecube_timestamp_range_sec, sleepstate_files[i], sleep_state_range_sec
                )
        sleep_state = np.concatenate(sleep_state_list)
        sleep_state_ecube_times = np.concatenate(sleep_state_ecube_times_list)
        assert np.all(np.diff(sleep_state_ecube_times) >= 0), \
            'sleep_state_ecube_times are not sorted, this condition should not occur under normal conditions.'
        sleep_state_ix = np.searchsorted(sleep_state_ecube_times, output_matrix['ecube_time'], side='left') - 1

        # Warn the user about 0 value sleep state labels
        count_zero_value_sleep_state = np.sum(sleep_state == 0)
        if count_zero_value_sleep_state > 0:
            print('Warning, {} sleep state labels are set to zero and are being treated as missing sleep state data (-1).'
                  .format(count_zero_value_sleep_state))

        # Any sleep_state_ix value of -1 is a frame before sleep state data is available (not valid).
        # Any sleep_state_ix value of sleep_state.shape[0] is data after sleep state data is available (not valid).
        # Any ecube_time values of 0 have no neural data to map sleep state to and are invalid.
        valid_ss = (sleep_state_ix >= 0) & (sleep_state_ix < sleep_state.shape[0]) & (output_matrix['ecube_time'] > 0) & \
                   (output_matrix['ecube_time'] <= sleep_state_ecube_times[-1] + int(4e+9))
        output_matrix['sleep_state'][valid_ss] = sleep_state[sleep_state_ix[valid_ss]]
        output_matrix['sleep_state'][~valid_ss] = -1
    else:
        output_matrix['sleep_state'] = -1

    #
    # Map in DLC labels, mapping 1:1 with video frames using the DLC & Video filename prefix
    #
    dlc_labels = {}
    output_matrix['dlc_label'] = -1.0  # represents unset DLC labels
    for f in dlclabel_files:
        k = os.path.split(f)[-1][:26]
        dlc = h5py.File(f, mode='r')['df_with_missing']['table']['values_block_0']
        dlc_labels[k] = np.concatenate((dlc_labels.get(k, np.empty(shape=(0, 6), dtype=dlc.dtype)), dlc), axis=0)

    for k, v in dlc_labels.items():
        video_file_ix = [i for i, v in enumerate(video_files) if os.path.split(v)[-1].startswith(k)]
        assert len(video_file_ix) == 1, 'DLC label {} did not match a single video filename as expected'.format(k)

        video_frames_mask = output_matrix['video_filename_ix'] == video_file_ix[0]
        n_video_frames = np.sum(video_frames_mask)
        n_dlc_labels = v.shape[0]

        # Validate that the number of DLC labels and video frames match, or ignore mismatches if ignore_dlc_mismatch
        if ignore_dlc_label_mismatch and np.sum(video_frames_mask) != v.shape[0]:
            print(
                'Warning, the number of DLC labels for {} is {}, but the number of video frames is {}'
                .format(k, v.shape[0], np.sum(video_frames_mask))
            )
            # override erroneous situation where more video frames exist than DLC labels
            video_frames_mask[np.where(video_frames_mask)[0][-(n_video_frames - n_dlc_labels):]] = False
        elif np.sum(video_frames_mask) != v.shape[0]:
            raise AssertionError(
                'The number of DLC labels for {} is {}, but the number of '
                'video frames is {}' .format(k, v.shape[0], np.sum(video_frames_mask))
            )

        output_matrix['dlc_label'][video_frames_mask] = v

    #
    # Verify all neural_files are represented in output_matrix, issue strong warning when they aren't.
    #
    missing_neural_files = np.setdiff1d(np.arange(len(neural_files)), output_matrix['neural_filename_ix'])
    for missing_neural_file_ix in missing_neural_files:
        print(
            f'***\n'
            f'*** Neural data file {neural_files[missing_neural_file_ix]} is not represented in output_matrix\n'
            f'***\n',
            end=None,
        )

    return output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files, dlclabel_files, \
        camera_pulse_output_matrix, pulse_ix


def _resolve_files(files: (list, tuple, str)):
    """ Resolves filename glob or string or list of filename strings to a static list of filenames. """
    files = [files] if isinstance(files, str) else files if files is not None else ()
    files = [_resolve_glob(f) for f in files]
    files = sorted(list(itertools.chain(*files)))
    return files


def _resolve_glob(file_glob):
    if file_glob.startswith('s3://'):
        _verify_s3_support()
        result = wr.s3.list_objects(file_glob)
    else:
        result = glob.glob(file_glob)

    return result


def _verify_s3_support():
    assert importlib.util.find_spec('awswrangler') and importlib.util.find_spec('smart_open') is not None, \
        'PIP packages awswrangler and smart_open are required for accessing ' \
        'S3 files: python -m pip install awswrangler smart_open'


def _resolve_neural_files_bom(neural_files_or_bom: list = None):
    """
    This function is typically used internally by map_video_to_neural_and_sleep_state(...),

    Use save_neural_files_bom to create the CSV once.

    This function resolves a list of neural filenames, or a CSV bill of materials containing a list of the neural
    files with their sizes and ecube timestamps to a list of (ecube_time, file_size, neural_filename)

    :param neural_files_or_bom: a list of neural files (non-globs), or a list of a single CSV file which is the
                                bill of materials (BOM) CSV file containing a list of all neural data files in format:
                                ecube_time, file_size, neural_filename
    :return: list in the form [(ecube_time, file_size, neural_filename), (...), ...]
    """
    assert neural_files_or_bom is not None and len(neural_files_or_bom) > 0, 'No neural files found.'
    uses_s3 = any([f.startswith('s3://') for f in neural_files_or_bom])
    if uses_s3:
        _verify_s3_support()

    if len(neural_files_or_bom) == 1 and neural_files_or_bom[0].endswith('.csv'):
        with smart_open.open(neural_files_or_bom[0], 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            result = [tuple(row) for row in csv_reader]
    else:
        # This process can take a long time when the files are on S3, parallelize
        with multiprocessing.Pool(20) as pool:
            result = pool.map(_read_ecube_time_and_file_size, neural_files_or_bom)

    return result


def _read_ecube_time_and_file_size(nfile):
    with smart_open.open(nfile, 'rb') as f:
        ecube_time = np.frombuffer(f.read(8), dtype=np.uint64, count=1)[0]
        file_size = wr.s3.size_objects(nfile)[nfile] if nfile.startswith('s3://') else os.fstat(f.fileno()).st_size
        filename = os.path.split(nfile)[-1]
        return ecube_time, file_size, filename


def save_neural_files_bom(output_filename: str = None, neural_files: (list, str) = None):
    """
    Example Usage:
        python ntk_sync.py save_neural_files_bom --output_filename neural_files_bom.csv --neural_files EAB40_Neural_data/*.bin
        python ntk_sync.py save_neural_files_bom --output_filename neural_files_bom.csv --neural_files s3://hengenlab/SCF05/Neural_Files/*.bin

    Saves a CSV bill of materials record of the neural_files. This eliminates needing to have the neural_files
    available when using the map_video_to_neural_and_sleep_state(...). Since the neural files are very large
    it may be inconvenient to make them available just to generate the mapping data.

    Note, this function uses an external package, you need to: `pip install fs_s3fs_ng`

    :param output_filename: output CSV filename for the bill of materials (BOM) file, default: neural_files_bom.csv
    :param neural_files: raw neural files, may be a list of filenames or globs or a string, both local files and
           s3 urls work (s3 example: 's3://BUCKET/path/*.bin), if using s3files you must install fs and fs_s3fs_ng
           pip packages.
    :return:
    """
    assert output_filename is not None
    assert neural_files is not None

    neural_files = _resolve_files(neural_files)
    bom = _resolve_neural_files_bom(neural_files)

    with open(output_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in bom:
            csv_writer.writerow(row)

def check_format_digital(digital): 
    if digital.shape[0] > 1:
        differences = np.diff(digital, axis=1)
        absolute = np.abs(differences)
        sums = np.sum(absolute, axis=1)
        index = np.argmax(sums)
    return digital[index]

def save_map_video_to_neural_data(output_filename: str = None,
                                  syncpulse_files: (str, list) = None,
                                  video_files: (str, list) = None,
                                  neural_files: (str, list) = None,
                                  sleepstate_files: (str, list) = None,
                                  dlclabel_files: (str, list) = None,
                                  fs: int = 25000,
                                  n_channels: int = None,
                                  neural_bin_files_per_sleepstate_file: int = None,
                                  manual_video_neural_offset_sec: int = 0,
                                  ignore_dlc_mismatch: bool = False,
                                  dataset_config: str = None,
                                  initial_neural_file_for_sleepstate_mapping: int = None):
    if dataset_config is not None:
        config_parser = configparser.ConfigParser()
        config_parser.read(dataset_config)
        config = config_parser['CONFIG']
        fs = int(config.get('fs', fs))
        n_channels = int(config.get('n_channels', n_channels))
        manual_video_neural_offset_sec = int(config.get('manual_video_frame_offset', manual_video_neural_offset_sec))
        ignore_dlc_mismatch = bool(distutils.util.strtobool((config.get('ignore_dlc_mismatch', ignore_dlc_mismatch))))
        neural_bin_files_per_sleepstate_file = int(config.get('neural_bin_files_per_sleepstate_file', 12))
        initial_neural_file_for_sleepstate_mapping = int(config.get('initial_neural_file_for_sleepstate_mapping'), 0)

    assert syncpulse_files is not None \
        and video_files is not None \
        and neural_files is not None \
        and fs is not None \
        and manual_video_neural_offset_sec is not None \
        and ignore_dlc_mismatch is not None \
        and neural_bin_files_per_sleepstate_file is not None, \
        'None value not valid.'

    output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files, \
        dlclabel_files, camera_pulse_output_matrix, pulse_ix \
        = map_video_to_neural_data(
            syncpulse_files=syncpulse_files, video_files=video_files, neural_files=neural_files,
            sleepstate_files=sleepstate_files, dlclabel_files=dlclabel_files, fs=fs, n_channels=n_channels,
            neural_bin_files_per_sleepstate_file=neural_bin_files_per_sleepstate_file,
            manual_video_neural_offset_sec=manual_video_neural_offset_sec, ignore_dlc_label_mismatch=ignore_dlc_mismatch,
            initial_neural_file_for_sleepstate_mapping=initial_neural_file_for_sleepstate_mapping
        )

    np.savez(
        file=output_filename, output_matrix=output_matrix, video_files=video_files, neural_files=neural_files,
        sleepstate_files=sleepstate_files, syncpulse_files=syncpulse_files, dlclabel_files=dlclabel_files,
        camera_pulse_output_matrix=camera_pulse_output_matrix, pulse_ix=pulse_ix
    )


def parse_args():
    parser_parent = argparse.ArgumentParser(description='Command line utilities for saving sync file data.')
    subparsers = parser_parent.add_subparsers(dest='command')
    subparsers.required = True

    # save_neural_files_bom
    parser_save_neural_file_bom = subparsers.add_parser(
        'save_neural_files_bom',
        help='Calls save_neural_files_bom(...) which saves just the eCube timestamp and file '
             'size for a set of neural data files, this can later be used in map_video_to_neural_data(...).'
    )
    parser_save_neural_file_bom.add_argument(
        '--output_filename', type=str, required=False, default='neural_files_bom.csv',
        help='Output CSV filename for the bill of materials (BOM) file, default: neural_files_bom.csv'
    )
    parser_save_neural_file_bom.add_argument(
        '--neural_files', type=str, action='append', required=True,
        help='Raw neural files, may a filename or glob, multiple entries are allowed, example: '
             '--neural_files file1 --neural_files file2*.bin'
    )

    # save_output_matrix
    parser_save_map_video_to_neural_data = subparsers.add_parser(
        'save_map_video_to_neural_data',
        help='Calls save_output_matrix(...) which saves the results of map_video_to_neural_data(...) to a NPZ file.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--output_filename', type=str, required=False, default='map_video_to_neural_data.npz',
        help='Output NPZ to contain all outputs of map_video_to_neural_data(...) by name.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--syncpulse_files', type=str, required=True, action='append',
        help='sync pulse filenames, may be a list or single string of filenames or globs.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--video_files', type=str, required=True, action='append',
        help='video filenames, may be a list or single string of filenames or globs.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--neural_files', type=str, required=True, action='append',
        help='raw neural files, may be a list or single string of filenames or globs, or a bill of materials '
             'CSV file that describes the set of neural datafiles'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--sleepstate_files', type=str, required=False, action='append',
        help='Optional sleep state numpy files, may be a list or single string of filenames or globs.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--dlclabel_files', type=str, required=False, action='append',
        help='Optional DLC label numpy files, may be a list or single string of filenames or globs.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--fs', type=int, required=False, default=25000,
        help='Pulse speed in ns, default=25000.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--n_channels', type=int, required=False, default=None,
        help='Number of channels in neural data, if unspecified the value will be inferred from the neural file names.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--manual_video_neural_offset_sec', type=float, required=False, default=0.0,
        help='See function documentation in function map_videoframes_to_syncpulse for details. in some cases the '
             'sync pulse doesn\'t properly identify the start of video recording. You can manually sync video to '
             'the neural data using file timestamps (see ntk documentation page for a guide on doing this). '
             'This value overrides the SyncPulse time relative to the start of neural data. Enter the video '
             'offset from neural data in seconds. In most cases video is started after neural data so this value '
             'will be positive, if video started before neural this value will be negative.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--neural_bin_files_per_sleepstate_file', type=int, default=12,
        help='Number of neural binary files that are covered by each sleep state '
             'numpy file. For example in EAB40 there are 12 neural files per sleep state file, therefore the '
             'parameter is 12; in SCF05 there is only one sleep state file for ALL neural data files, in this case the '
             'parameter is -1 for ALL.'
    )
    parser_save_map_video_to_neural_data.add_argument(
        '--initial_neural_file_for_sleepstate_mapping', type=int, default=0,
        help='If sleep state data doesn''t map to the first neural file present in the dataset, set this '
             'parameter to the number of neural files to skip before mapping to the sleep state data. '
             'For example, EAB50 maps sleep state to location 228 (0 indexed), which is file: '
             'Headstages_512_Channels_int16_2019-06-21_12-05-11.bin'
    )
    return vars(parser_parent.parse_args())


if __name__ == '__main__':
    args = parse_args()                     # Parse command line arguments
    func = globals()[args.pop('command')]   # Get the function name from args['command']
    func(**args)                            # Call the appropriate function passing the rest of the args
