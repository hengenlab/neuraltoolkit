#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Script to post process dlc outputs

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

List of functions/class in ntk_ecube
dlc_get_position(videoh5file, cutoff=0.6)
'''

try:
    import numpy as np
except ImportError:
    raise ImportError('Run command : conda install numpy')
try:
    import pandas as pd
except ImportError:
    raise ImportError('Run command : conda install pandas')
import glob
import os.path as op
import shutil
import matplotlib.pyplot as plt
import itertools
from neuraltoolkit import load_digital_binary_allchannels
from neuraltoolkit import load_digital_binary


# Position
def dlc_get_position(videoh5file, cutoff=0.6):

    '''
    Get x,y postion of features and names from h5 output of deeplabcut

    def dlc_get_position(videoh5file, cutoff)

    Parameters
    ----------
    videoh5file : File name of h5 file with path
    cutoff : based on confidence (default=0.6)

    Returns
    -------
    positions_h5 : x, y position of labels in video
    feature_names : name of the features

    Raises
    ------

    See Also
    --------

    Notes
    -----

    Examples
    --------
    cutoff = 0.6
    vidoh5_file = \
        'D17_trial1DeepCut_resnet50_crickethuntJul18shuffle1_15000.h5'
    pos, fnames = dlc_get_position(vidoh5_file, cutoff)

    '''

    # Load h5 file
    df = pd.read_hdf(videoh5file)

    # Get feature names
    feature_names = []
    feature_names_list = df.columns.values[:]
    for _ in range(0, len(feature_names_list), 3):
        feature_names.append(feature_names_list[_][1])

    # Get x, y and likelihood for all features
    features_xylikelihood = df.values

    # Calculate number of features
    nfeatures = int(np.shape(features_xylikelihood)[1]/3)
    print("Number of features", nfeatures)

    # Loop over feature and change low likelihood x, y values to np.nan
    for i in range(0, nfeatures):
        pro = features_xylikelihood[:, 3*i+2]
        lowpro = np.where(pro < cutoff)[0]
        features_xylikelihood[lowpro, 3*i] = np.nan
        features_xylikelihood[lowpro, 3*i+1] = np.nan

    # remove likelihood
    positions_h5 = np.delete(features_xylikelihood,
                             np.arange(2, nfeatures*3, 3), axis=1)
    return positions_h5, feature_names


def find_video_start_index(datadir, ch, nfiles=10,
                           fs=25000, fps=15,
                           lnew=1,
                           fig_indx=None):
    '''
    find_video_start_index(datadir, ch, nfiles=10,
                           fs=25000, fps=30,
                           fig_indx=None)

    datadir: data directory where digital file is located
    ch : channel where Watchtower signal is recorded,
         remember number starts from 0
    nfiles: First how many files to check for pulse change
        (default first 10 files)
    fs: Sampling rate of digital file (default 25000)
    fps: Frames per second of video file (default 15)
    lnew: default 1, new digital files.
          0 for old digital files with only one channel
    fig_indx: Default None, if index is given it plot figure
    '''

    # check datadir exists
    if not (op.exists(datadir) and op.isdir(datadir)):
        raise NotADirectoryError("Directory {} does not exists".
                                 format(datadir))

    fl_list = np.sort(glob.glob(datadir + op.sep + 'Dig*.bin'))
    if len(fl_list) == 0:
        raise ValueError("No digital files found in ", datadir)
    print("fig_indx ", fig_indx, " ", datadir)
    if fig_indx is not None:
        plt.figure(num=fig_indx, figsize=(20, 4))
        plt.title(datadir)
    cumulative_len = 0
    for indx in range(nfiles):
        if lnew == 1:
            t, data = load_digital_binary_allchannels(fl_list[indx],
                                                      channel=ch)
        elif lnew == 0:
            t, data = load_digital_binary(fl_list[indx], t_only=0,
                                          lcheckdigi64=1)
        data_group_list = None
        data_group_list = []
        for _, data_group in itertools.groupby(data):
            data_group_list.append(len(list(data_group)))

        try:
            # video_start_group_index = \
            #     np.where(np.asarray(data_group_list) ==
            #              int(fs/(fps*2)))[0][0]
            video_start_group_index = \
                np.where((np.asarray(data_group_list) == int(fs/(fps*2))) |
                         (np.asarray(data_group_list) == (int(fs/(fps*2))+1))
                         )[0][0]
            video_start_index = \
                np.sum(data_group_list[0:video_start_group_index])
            if fig_indx is not None:
                plt.plot(np.arange(video_start_index-10000,
                         video_start_index+10000, 1),
                         data[video_start_index - 10000:
                              video_start_index + 10000],
                         label=str(fl_list[indx]))
                plt.plot([video_start_index]*5,
                         [0, 0.25, 0.50, 0.75, 1.00],
                         'g*',
                         linewidth=20,
                         markersize=20)
                plt.legend()
                plt.show()
            print("found in ", fl_list[indx], flush=True)
            return video_start_index + cumulative_len
            break
        except Exception as e:
            # ugly
            if 0:
                print("Error ", e)
            cumulative_len = cumulative_len + data.shape[0]
            # print(cumulative_len)
            # print("not found in ", str(fl_list[indx]))
            # print(indx)
            if (indx + 1) == nfiles:
                raise\
                    RuntimeError("Try increasing nfile or check fps or ch")


def append_emptyframes_todlc_h5file(h5_file_name, video_fps, target_rows=None):
    '''
    Append missing empty row to dlc output h5 file to make it an hour
    This can be used if video is stopped just short of 1hour and you have
    neuralrecording for whole hour

    def append_emptyframes_todlc_h5file(h5_file_name, video_fps):

    Parameters
    ----------
    h5_file_name : File name of h5 file with path
    video_fps: The sampling rate of the video (frames per second).
    target_rows: The target number of rows to add. If None (default),
        the number of rows is determined by video_fps. If specified,
        this value takes precedence over video_fps,
        and the number of rows will be calculated accordingly.

    Returns
    -------
    create copy of h5_file_name as h5_file_name_back
    make a new h5_file_name with frames for 1 hour

    Raises
    ------
    FileNotFoundError : if h5_file_name does not exist

    See Also
    --------

    Notes
    -----

    Examples
    --------
    video_fps = 15
    h5_file_name = \
     '/home/kiran/D17_trial1DeepCut_resnet50_crickethuntJul18shuffle1_15000.h5'
    ntk.append_emptyframes_todlc_h5file(h5_file_name, video_fps)


    '''
    # Check if the file exists
    if not op.isfile(h5_file_name):
        raise FileNotFoundError(f"The file {h5_file_name} does not exist.")

    # Make a copy of the h5 file
    backup_file_name = f"{h5_file_name}_back"
    # os.system(f"cp {h5_file_name} {backup_file_name}")
    # Using shutil to copy the file
    shutil.copy2(h5_file_name, backup_file_name)

    # Load the HDF5 file
    df = pd.read_hdf(h5_file_name)

    # Make a copy of the DataFrame
    dfnew = df.copy()

    # Calculate the target number of rows based on video fps
    if target_rows is None:
        target_rows = video_fps * 3600
        rows_to_add = target_rows - dfnew.shape[0]

    # If additional rows are needed
    if rows_to_add > 0:
        # Create an empty DataFrame with the same columns
        empty_rows = pd.DataFrame(index=range(rows_to_add),
                                  columns=dfnew.columns)

        # Remove all-NA columns from empty_rows (if any)
        empty_rows = empty_rows.dropna(axis=1, how='all')

        # Concatenate the original DataFrame with the empty DataFrame
        dfnew = pd.concat([dfnew, empty_rows], ignore_index=True)

    # Replace the original h5 file with the new DataFrame
    dfnew.to_hdf(h5_file_name, key='df', mode='w')
    print(f"Added {rows_to_add} rows to {h5_file_name}.")
