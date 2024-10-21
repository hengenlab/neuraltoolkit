#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to do basic math

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


List of functions/class in ntk_math
data_intpl(tvec, dvec, nfact, intpl_kind='cubic')
'''


import numpy as np
try:
    from scipy import interpolate
except ImportError:
    raise ImportError('Run command : conda install scipy')


def data_intpl(tvec, dvec, nfact, intpl_kind='cubic', verbose=False):
    '''
    Interpolates the data vector `dvec` by increasing the number of samples
     in `tvec` by a factor of `nfact`.

    Parameters:
    -----------
    tvec : array-like
        Time or sample vector representing the independent variable.
    dvec : array-like
        Data vector representing the dependent variable corresponding
         to `tvec`.
    nfact : int
        The interpolation factor by which the number of points should be
         increased.
    intpl_kind : str, optional
        Type of interpolation to perform (default is 'cubic').
        Available options:
          - 'linear'     : linear interpolation between points.
          - 'nearest'    : nearest neighbor interpolation.
          - 'zero'       : piecewise constant interpolation (0th-order spline).
          - 'slinear'    : first-order spline interpolation.
          - 'quadratic'  : second-order spline interpolation.
          - 'cubic'      : third-order spline interpolation (default).
    verbose : bool, optional
        If True, prints out the details of the interpolation process
        (default is False).

    Returns:
    --------
    tvect_intpl : ndarray
        New time/sample vector with interpolated points, increased by `nfact`.
    dvec_intpl : ndarray
        Interpolated data corresponding to `tvect_intpl`.

    Raises:
    -------
    ValueError:
        If `nfact` is less than 1 or if `tvec` and `dvec` are not of
         equal length.
        If `intpl_kind` is invalid.

    Example:
    --------
    tvec = np.array([0, 1, 2, 3])
    dvec = np.array([0, 1, 0, -1])
    nfact = 2

    tvec_intpl, dvec_intpl = data_intpl(tvec, dvec, nfact)
    '''

    # Convert input lists or tuples to numpy arrays
    tvec = np.asarray(tvec)
    dvec = np.asarray(dvec)

    # Validate the input
    if tvec.shape[0] != dvec.shape[0]:
        raise ValueError("tvec and dvec must have the same length.")
    if nfact < 1:
        raise ValueError("nfact must be a positive integer.")

    valid_interpolations = \
        ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    if intpl_kind not in valid_interpolations:
        raise ValueError(f"Invalid {intpl_kind}, use {valid_interpolations}")

    # Handle the single-point case
    if len(tvec) == 1:
        if verbose:
            print("Single-point input detected. Returning the point itself.")
        return tvec, dvec

    # Number of points to interpolate
    npoints = tvec.shape[0] * nfact
    if verbose:
        print(f"Interpolating using {intpl_kind} with {npoints} points")

    # Create the new interpolated time vector
    tvect_intpl = np.linspace(tvec[0], tvec[-1], npoints)

    # Perform the interpolation
    data_intpl = \
        interpolate.interp1d(tvec, dvec, kind=intpl_kind,
                             fill_value="extrapolate")

    # Return the interpolated time and data
    dvec_intpl = data_intpl(tvect_intpl)

    if verbose:
        print("Interpolation completed successfully.")

    return tvect_intpl, dvec_intpl
