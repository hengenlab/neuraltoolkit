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
