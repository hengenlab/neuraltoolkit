""" Tools for syncing between video, neural, and sleep states. """
import glob
import itertools
import os
import numpy as np
import neuraltoolkit as ntk


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
    ecube_start_time = None
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
        t, dr = ntk.load_digital_binary(df)

        if i == 0:
            assert dr[0] == 0, 'This algorithm expects the first value of the first file to always be 0.'
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
    cumsum_files = np.cumsum([np.sum(x) for x in all_file_lengths])
    cumsum_files = np.insert(cumsum_files, 0, 0)[:-1]

    output_matrix['binfile_ix'] = np.searchsorted(cumsum_files, cumsum_signal_widths, side='right') - 1
    output_matrix['offsets'] = cumsum_signal_widths - cumsum_files[output_matrix['binfile_ix']]
    # See function docs to understand why ecube_time is only computed relative to the first file and not from all files.
    output_matrix['ecube_time'] = ecube_start_time + cumsum_signal_widths * ecube_interval

    return output_matrix, pulse_ix, files


def map_video_to_neural_and_sleep_state(syncpulse_files: (list, str),
                                        video_files: (list, str),
                                        neural_files: (list, str),
                                        sleepstate_files: (list, str),
                                        fs=25000, n_channels=256):
    """
    EXAMPLE USAGE:
        output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files = \
            ntk.sync_video_to_neural_and_sleep_state(
                syncpulse_files='EAB40Data/EAB40_Camera_Sync_Pulse/*.bin'
                video_files=['EAB40Data/EAB40_Video/3_29-4_02/*.mp4',
                             'EAB40Data/EAB40_Video/4_02-4_05/*.mp4'],
                neural_files='EAB40Data/EAB40_Neural_Data/3_29-4_02/*.bin',
                sleepstate_files='EAB40Data/EAB40_Sleep_States/*.npy'
            )

    Output matrix format:
    [(-1,       0,   0,     0,  -1,     -1,     430271801375)  <- First video frame has no sleep state or neural data
     (-1,       1,   0,     1,  -1,     -1,     430338481375)
     (-1,       2,   0,     2,  -1,     -1,     430405121375)
     (-1,       3,   0,     3,  -1,     -1,     430471801375)
     ...
     (-1,    4243,   0,  4243,  -1,     -1,     713106641375)
     (-1,    4244,   0,  4244,  -1,     -1,     713173321375)
     ( 3,    4245,   0,  4245,   0,   1272,     713239961375)  <- Video frame 4245, first frame with neural & sleep
     ( 3,    4246,   0,  4246,   0,   2939,     713306641375)
     ( 3,    4247,   0,  4247,   0,   4606,     713373321375)
     ...
     ( 1, 1300345,  37,  4435, 287, 7495704,  87119027721375)
     ( 1, 1300346,  37,  4436, 287, 7497370,  87119094361375)
     ( 1, 1300347,  37,  4437, 287, 7499037,  87119161041375)  <- Last video frame with neural & sleep data
     (-1, 1300348,  37,  4438,  -1,      -1,  87119227721375)  <- Video frames beyond neural & sleep data (-1)
     (-1, 1300349,  37,  4439,  -1,      -1,  87119294361375)
     (-1, 1300350,  37,  4440,  -1,      -1,  87119361041375)
     ...
     (-1, 4870488, 109, 53996,  -1,      -1, 325126320121375)
     (-1, 4870489, 109, 53997,  -1,      -1, 325126386761375)
     (-1, 4870490, 109, 53998,  -1,      -1, 325126453441375)
      --  -------  ---  -----  ---  -------  ---------------
      (a)   (b)    (c)   (d)   (e)    (f)         (g)

    (a) sleep_state           - Sleep state (1=wake, 2=NREM, 3=REM)
    (b) video_frame_global_ix - The video frame index (0 based) counting across all video files (a global frame counter)
    (c) video_filename_ix     - Index to the video filename returned in video_files
    (d) video_frame_offset    - The video frame index (0 based) counting from the current file
    (e) neural_filename_ix    - Index to the raw neural filename returned in neural_files
    (f) neural_offset         - Sample offset in the neural file reference in neural_filename_ix
                                Note that this is not an index into the file directly, to get the index to the file use:
                                (offset * n_channels * 2 + 8), int16 values plus the 8-byte eCube timestamp offset
    (g) ecube_time            - The exact ecube time when this frame began the write process

    Detailed explanation of a few entries above:
    (-1,       0,   0,     0,  -1,     -1,     430271801375)
        (-1) tells us there is no sleep data for this video frame; (0) is the global video frame index (counting across
        all video files); (0) is the index into the video filename returned by video_files; (0) is the index of the
        video frame within the current file only; (-1) and (-1) tell us that there is no neural data associated with
        this video frame (in this case the video started before neural recording); and (430271801375) is the eCube
        timestamp when this video file began writing.
    ( 3,    4245,   0,  4245,   0,   1272,     713239961375)
        (3) is the sleep state, REM sleep; (4245) is the global video frame index (counting across all video files);
        (0) tells us this frame came from the first video file returned in video_files; (4245) is the index of the
        frame counting from the current video file (the same as the global counter for the first video file naturally);
        (0) is an index to the raw neural data file returned in neural_files; (1272) is the offset of the data samples
        in the raw neural data file (remember this isn't the file offset, to calculate the actual raw file offset
        compute: 1272 * n_channels* 2 + 8); (713239961375) is the eCube timestamp when the video frame started writing.
    ( 1, 1300347,  37,  4437, 287, 7499037,  87119161041375)
        (1) is the sleep state, awake; (1300347) is the global video frame index (counting across all video files);
        (37) is an index to the video file returned in video_files; (4437) is the frame index counting from the current
        video file; (287) is the index to the neural data file returned by neural_files; (7499037) is the offset
        of the data samples in the raw neural data file (remember this isn't the file offset, to compute the actual raw
        file offset compute: 7499037 * n_channels * 2 + 8); (87119161041375) ) is the eCube timestamp when the video
        frame started writing.
    (-1, 1300348,  37,  4438,  -1,      -1,  87119227721375)
        (-1) tells us there is no sleep data for this video frame; (1300348) is the global video frame index
        (counting across all video files); (37) is an index to the video file returned in video_files; (4438) is
        the frame index counting from the current video file; (-1) and (-1) tell us that there is no neural data
        associated with this video frame (in this example the video files continued beyond the neural recording data).

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

    :param syncpulse_files: sync pulse filenames, may be a list of filenames or globs or a string.
    :param video_files: video filenames, may be a list of filenames or globs or a string.
    :param neural_files: raw neural files, may be a list of filenames or globs or a string.
    :param sleepstate_files: sleep state numpy files, may be a list of filenames or globs or a string.
    :param fs: sample rate, default = 25000
    :param n_channels: number of channels in data files, default = 256
    :return: output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files
    """
    # Resolve all inputs to lists of files
    syncpulse_files = syncpulse_files if isinstance(syncpulse_files, list) else [syncpulse_files]
    syncpulse_files = [glob.glob(sfg) for sfg in syncpulse_files]
    syncpulse_files = sorted(list(itertools.chain(*syncpulse_files)))
    video_files = video_files if isinstance(video_files, list) else [video_files]
    video_files = [glob.glob(vfg) for vfg in video_files]
    video_files = sorted(list(itertools.chain(*video_files)))
    neural_files = neural_files if isinstance(neural_files, list) else [neural_files]
    neural_files = [glob.glob(nfg) for nfg in neural_files]
    neural_files = sorted(list(itertools.chain(*neural_files)))
    sleepstate_files = sleepstate_files if isinstance(sleepstate_files, list) else [sleepstate_files]
    sleepstate_files = [glob.glob(sfg) for sfg in sleepstate_files]
    sleepstate_files = sorted(list(itertools.chain(*sleepstate_files)))
    assert len(syncpulse_files) > 0 and len(video_files) > 0 and len(neural_files) > 0 and len(sleepstate_files) > 0

    # output_matrix data types definition
    structured_array_dtypes = [
        ('sleep_state', np.int8),
        ('video_frame_global_ix', np.uint32),
        ('video_filename_ix', np.int32),
        ('video_frame_offset', np.uint32),
        ('neural_filename_ix', np.int32),
        ('neural_offset', np.int64),
        ('ecube_time', np.uint64),
    ]
    ecube_interval = 1000000000 // fs

    neural_ecube_timestamps = []
    neural_sample_counts = []

    # Count frames in all video files
    frame_counts = np.array([round(ntk.NTKVideos(vfile, 0).length) for vfile in video_files])
    total_frames = np.sum(frame_counts)

    # Extract just the eCube timestamp from each raw neural recording file
    for nfile in neural_files:
        with open(nfile, 'rb') as f:
            neural_ecube_timestamps.append(np.fromfile(f, dtype=np.uint64, count=1))
            neural_sample_counts.append((os.fstat(f.fileno()).st_size - 8) // (2 * n_channels))
    neural_ecube_timestamps = np.concatenate(neural_ecube_timestamps, axis=0)
    neural_sample_counts = np.array(neural_sample_counts)
    neural_ecube_last_timestamp = \
        neural_ecube_timestamps[-1] + (ecube_interval * neural_sample_counts[-1]).astype(np.uint64)

    # Validate eCube timestamps are correct, issue a warning if they are significantly outside expectation
    # This is necessary because eCube timestamps have been observed to be erroneous.
    validate_neural_ecube_timestamps = np.abs(
        (neural_sample_counts[:-1] * ecube_interval) -
        (neural_ecube_timestamps[1:] - neural_ecube_timestamps[:-1]).astype(np.int64)
    )
    if np.any(np.abs(validate_neural_ecube_timestamps) > ecube_interval * 3):
        for i in np.where(validate_neural_ecube_timestamps > ecube_interval * 3)[0]:
            print(
                'WARNING: eCube timestamp deviation of {:.2f} seconds for file {}, the eCube timestamp does not match '
                'expectation based on the eCube timestamp of the previous file, eCube time of this file is {:d}, '
                'and the previous eCube timestamp is {:d}.'
                .format(validate_neural_ecube_timestamps[i] / 1000000000,
                        neural_files[i],
                        neural_ecube_timestamps[1:][i],
                        neural_ecube_timestamps[i])
            )
        raise AssertionError('Invalid eCube timestamp detected, this will cause errors in indexing.')

    output_matrix = np.empty((total_frames,), dtype=structured_array_dtypes)

    # Compute a precise eCube time per video frame
    cumsum_frame_counts = np.insert(np.cumsum(frame_counts), 0, 0)[:-1]
    camera_pulse_output_matrix, pulse_ix, _ = map_videoframes_to_syncpulse(syncpulse_files)
    output_matrix['ecube_time'] = camera_pulse_output_matrix['ecube_time'][pulse_ix::2][:total_frames]

    output_matrix['video_frame_global_ix'] = np.arange(0, total_frames)
    output_matrix['video_filename_ix'] = \
        np.searchsorted(cumsum_frame_counts, output_matrix['video_frame_global_ix'], side='right') - 1
    output_matrix['video_frame_offset'] = \
        output_matrix['video_frame_global_ix'] - cumsum_frame_counts[output_matrix['video_filename_ix']]

    # Compute the neural file index and offset for that file, noting that some frames will be recorded before neural
    # data recording started, in those cases the neural_filename_ix and neural_offset will be -1
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

    # Map in sleep states based on eCube time
    neural_ecube_timestamps_per_sleep_state_file = neural_ecube_timestamps[::12]
    sleep_state_list = [np.load(sfile) for sfile in sleepstate_files]
    assert len(neural_ecube_timestamps_per_sleep_state_file) > len(sleep_state_list), \
        'There are more sleep state files than neural recording files.'
    sleep_state_ecube_times_list = [
        np.linspace(start=neural_ecube_timestamps_per_sleep_state_file[i],
                    stop=neural_ecube_timestamps_per_sleep_state_file[i + 1],
                    num=sleep_state_list[i].shape[0], dtype=np.uint64)
        for i in range(len(sleep_state_list))
    ]
    sleep_state_ecube_times_list.append(np.array([sleep_state_ecube_times_list[-1][-1] + 4000000000], dtype=np.uint64))

    sleep_state = np.concatenate(sleep_state_list)
    sleep_state_ecube_times = np.concatenate(sleep_state_ecube_times_list)
    sleep_state_ix = np.searchsorted(sleep_state_ecube_times, output_matrix['ecube_time'], side='left') - 1
    # Any sleep_state_ix value of -1 is a frame before sleep state data is available (not valid),
    # Any sleep state_ix value of sleep_state.shape[0] is data after sleep state data is available (not valid).
    valid_ss = np.logical_and(sleep_state_ix >= 0, sleep_state_ix < sleep_state.shape[0])
    output_matrix['sleep_state'][valid_ss] = sleep_state[sleep_state_ix[valid_ss]]
    output_matrix['sleep_state'][~valid_ss] = -1

    return output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files
