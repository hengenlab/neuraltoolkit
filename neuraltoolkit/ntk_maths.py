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


def data_intpl(tvec, dvec, nfact, intpl_kind='cubic'):

    '''
    take time/number of samples and data and interpolate by nfact times
    tvec - time/number of samples
    dvec - data vector
    nfact - number of points to increase
    intpl_kind - type of interpolation (Default 'cubic')
      'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
       where 'zero', 'slinear', 'quadratic' and 'cubic' refer to order of
       spline interpolation


    returns tvec_intpl and dvec_intpl, intepolated data
    '''

    npoints = tvec.shape[0] * nfact
    # print("npoints is", npoints)
    tvect_intpl = np.linspace(tvec[0], tvec[-1], npoints)

    data_intpl = interpolate.interp1d(tvec, dvec, kind=intpl_kind)

    return tvect_intpl, data_intpl(tvect_intpl)
