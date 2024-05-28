import numpy as np
import time
import matplotlib.pyplot as plt
import os.path as op
from datetime import datetime, timedelta


def remove_large_noise(ddgc_filt, max_value_to_check=3000,
                       windowval=500, checkchans=None,
                       lplot=0, lverbose=0):

    '''
    remove_large_noise(ddgc_filt, max_value_to_check=2500,
                       windowval=500, checkchans=None,
                       lplot=0)

    ddgc_filt: raw data matrix
    max_value_to_check : max value to remove (default 2500)
    windowval : window to zero (default 500)
    checkchans : list of 4 channels (default None)
    lplot : 1 to plot first channel from checkchans (default 0)

    returns
    data: cleaned up ddgc_filt
    indices : where data is above max_value_to_check


    raise ValueError
    '''

    if (len(checkchans) < 1):
        raise ValueError(f'Error: checkchans length < 1: {len(checkchans)}')
    if (len(checkchans) > 4):
        raise ValueError(f'Error: checkchans length > 4: {len(checkchans)}')
    if lverbose:
        print(f'plotting raw data {lplot}', flush=True)

    ddgc = ddgc_filt * 1

    # max_value_to_check = 2500
    # width_to_check = 1
    # windowval = 500

    # Artifact checks
    otic = time.time()
    edges = None
    edges1 = None
    edges2 = None
    edges3 = None
    edges4 = None

    if len(checkchans) == 1:
        edges1 = np.argwhere(np.abs(ddgc[checkchans[0], :]) >
                             max_value_to_check).flatten()
        edges1 = np.array(edges1, dtype='int')
    elif len(checkchans) == 2:
        edges1 = np.argwhere(np.abs(ddgc[checkchans[0], :]) >
                             max_value_to_check).flatten()
        edges1 = np.array(edges1, dtype='int')
        edges2 = np.argwhere(np.abs(ddgc[checkchans[1], :]) >
                             max_value_to_check).flatten()
        edges2 = np.array(edges2, dtype='int')
    elif len(checkchans) == 3:
        edges1 = np.argwhere(np.abs(ddgc[checkchans[0], :]) >
                             max_value_to_check).flatten()
        edges1 = np.array(edges1, dtype='int')
        edges2 = np.argwhere(np.abs(ddgc[checkchans[1], :]) >
                             max_value_to_check).flatten()
        edges2 = np.array(edges2, dtype='int')
        edges3 = np.argwhere(np.abs(ddgc[checkchans[2], :]) >
                             max_value_to_check).flatten()
        edges3 = np.array(edges3, dtype='int')
    elif len(checkchans) == 4:
        edges1 = np.argwhere(np.abs(ddgc[checkchans[0], :]) >
                             max_value_to_check).flatten()
        edges1 = np.array(edges1, dtype='int')
        edges2 = np.argwhere(np.abs(ddgc[checkchans[1], :]) >
                             max_value_to_check).flatten()
        edges2 = np.array(edges2, dtype='int')
        edges3 = np.argwhere(np.abs(ddgc[checkchans[2], :]) >
                             max_value_to_check).flatten()
        edges3 = np.array(edges3, dtype='int')
        edges4 = np.argwhere(np.abs(ddgc[checkchans[3], :]) >
                             max_value_to_check).flatten()
        edges4 = np.array(edges4, dtype='int')

    if len(checkchans) == 1:
        edges = np.unique(edges1)
    elif len(checkchans) == 2:
        edges = np.unique(np.concatenate((edges1, edges2)))
    elif len(checkchans) >= 3:
        edges = np.unique(np.concatenate((edges1, edges2, edges3)))
    elif len(checkchans) >= 4:
        edges = np.unique(np.concatenate((edges1, edges2, edges3, edges4)))
    # print(f'shape edges {edges.shape}', flush=True)

    edges1 = None
    edges2 = None
    edges3 = None
    edges4 = None
    del edges1
    del edges2
    del edges3
    del edges4

    edges_all = None
    # edges_all = []
    edges_all = np.array([], dtype='int')
    last_i = -ddgc.shape[1] + 100
    for i in edges:
        if lverbose:
            print(f'i {i} last_i {last_i}')
            print(f'(i - last_i) {(i - last_i)}')
            print(f'windowval//2 {windowval//2}')
            print(f'(i - last_i) > windowval//2 {(i - last_i) > windowval//2}')
        if (i - last_i) > windowval//2:
            edges_all = np.concatenate((edges_all,
                                       np.arange(min(i,
                                                     (i - windowval)),
                                                 min(ddgc.shape[1],
                                                     (i + windowval)), 1)))
            last_i = i
    if lverbose:
        print('edges_all done', flush=True)
    edges_all = np.sort(np.unique(edges_all))
    ddgc[:, edges_all] = 0

    otoc = time.time()
    if lverbose:
        print(f'Artifact removal took {otoc - otic} seconds')
        print(f'len edges_all {len(edges_all)}', flush=True)
    if len(edges) < 1:
        lplot = 0

    if lplot == 1:
        fig, ax = plt.subplots(nrows=3, figsize=(40, 6), sharex=True)
        fig.suptitle(f'Channel {checkchans[0]}')
        for indx, ch in enumerate(checkchans[0:1]):
            if lverbose:
                print(f'indx {indx} ch {ch}')
                print(f'edges {edges}', flush=True)
            ax[3*indx].plot(ddgc_filt[ch, :])
            ax[(3*indx) + 1].plot(ddgc[ch, :])
            ax[(3*indx) + 2].plot(edges_all, np.arange(len(edges_all)))
            ax[(3*indx) + 2].set_xlim([0, len(ddgc[ch, :])])
        plt.tight_layout()
        plt.show()
    elif lplot != 0:
        if op.exists(op.dirname(lplot)):
            if lverbose:
                print(f'plotting raw data {lplot}', flush=True)
                print(f'op.dirname(lplot) {op.dirname(lplot)}', flush=True)
            fig, ax = plt.subplots(nrows=3, figsize=(40, 6), sharex=True)
            fig.suptitle(f'Channel {checkchans[0]}')
            for indx, ch in enumerate(checkchans[0:1]):
                if lverbose:
                    print(f'indx {indx} ch {ch}')
                    print(f'edges {edges}', flush=True)
                ax[3*indx].plot(ddgc_filt[ch, :])
                ax[(3*indx) + 1].plot(ddgc[ch, :])
                ax[(3*indx) + 2].plot(edges_all, np.arange(len(edges_all)))
                ax[(3*indx) + 2].set_xlim([0, len(ddgc[ch, :])])
            plt.tight_layout()
            # plt.show()
            plt.savefig(lplot)

    plt.close('all')
    return ddgc, edges_all


def generate_filenames_in_ecubeformat(initial_filename, total_minutes=256,
                                      interval_minutes=5):

    '''
    generate_filenames_in_ecubeformat(initial_filename, total_minutes=256,
                                      interval_minutes=5)
    This is used to split a large ecube files to 5 minutes files.

    initial_filename : Initial headstage filename same as ecube format
     'Headstages_64_Channels_int16_2023-12-19_13-28-09.bin'

    total_minutes : total minutes we need to create file names for.

    interval_minutes : In minutes, default 5 minutes

    '''

    # Extract the base part and the time part of the filename
    base_part = '_'.join(initial_filename.split('_')[:-1])
    time_part = initial_filename.split('_')[-1].split('.')[0]

    # Create a datetime object from the time part
    time_format = '%H-%M-%S'
    current_time = datetime.strptime(time_part, time_format)

    # List to hold all the filenames
    ecube_filenames = None
    ecube_filenames = []

    for _ in range(int((total_minutes / interval_minutes) + 1)):
        # Update the time for the next iteration
        current_time += timedelta(minutes=interval_minutes)

        # Generate the new filename
        new_filename = f"{base_part}_{current_time.strftime(time_format)}.bin"
        ecube_filenames.append(new_filename)

    return ecube_filenames


def get_tetrode_channels_from_channelnum(channel1, ch_grp_size=4):
    """
    Returns a list of channels in the same tetrode as the given channel number.

    get_tetrode_channels_from_channelnum(channel1, ch_grp_size=4)
    channel1 (int): The channel number in the tetrode.
    ch_grp_size (int, optional): The number of channels per tetrode.
     Defaults to 4.

    returns
    list: A list of channel numbers in the same tetrode.

    Raises:
    ValueError: If channel1 is less than 0.
    """
    if channel1 < 0:
        raise ValueError("channel1 must be a non-negative integer")

    start_channel = channel1 - (channel1 % ch_grp_size)
    end_channel = start_channel + ch_grp_size
    ch_list = np.arange(start_channel, end_channel).tolist()
    ch_list = [int(i) for i in ch_list]
    return ch_list


def get_tetrode_channels_from_tetnum(tetrode_num, ch_grp_size=4):
    """
    Returns a list of channel numbers for the given tetrode number.

    get_tetrode_channels_from_tetnum(tetrode_num, ch_grp_size=4)
    tetrode_num (int): The tetrode number (must be between 0 and 15 inclusive).
    ch_grp_size (int, optional): The number of channels per tetrode.
       Defaults to 4.

    Returns:
    list: A list of channel numbers corresponding to the tetrode number.

    Raises:
    ValueError: If tetrode_num is not between 0 and 15 inclusive.

    Examples:
    >>> get_tetrode_channels_from_tetnum(0)
    [0, 1, 2, 3]

    >>> get_tetrode_channels_from_tetnum(1)
    [4, 5, 6, 7]

    >>> get_tetrode_channels_from_tetnum(15)
    [60, 61, 62, 63]

    >>> get_tetrode_channels_from_tetnum(0, ch_grp_size=5)
    [0, 1, 2, 3, 4]
    """
    if tetrode_num < 0 or tetrode_num > 15:
        raise ValueError("tetrode_num must be between 0 and 15 inclusive")

    start = tetrode_num * ch_grp_size
    end = start + ch_grp_size
    return list(range(start, end))
