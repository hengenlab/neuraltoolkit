#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Functions for plotting

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
'''

from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects
# from rpy2.robjects.vectors import StrVector
from rpy2.rinterface_lib.embedded import RRuntimeError
import math


def ntk_calculate_cohens_d(mean1, mean2, std1, std2):
    """
    Calculate Cohen's d.
    def ntk_calculate_cohens_d(mean1, mean2, std1, std2):

    Parameters:
    mean1, mean2 : means of groups 1 and 2
    std1, std2 : standard deviations of groups 1 and 2

    Returns:
    Cohen's d as a float
    """

    # Calculate the pooled standard deviation
    pooled_std = math.sqrt((math.pow(std1, 2) + math.pow(std2, 2)) / 2)

    # Calculate Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std

    return cohens_d


def ntk_power_t_test(n=None, d=None, sig_level=0.05,
                     power=0.8,
                     type="two.sample",
                     alternative="two.sided"):

    '''
    def ntk_power_t_test(n=None, d=None, sig_level=0.05,
                        power=0.8,
                        type="two.sample",
                        alternative="two.sided"):

    n :  Number of observations (per sample)

    d :  Effect size (Cohen's d) - difference between the means
        divided by the pooled standard deviation

    sig.level :  Significance level (Type I error probability)

    power :  Power of test (1 minus Type II error probability)

    type :  Type of t test : one- two- or paired-samples

    alternative :  a character string specifying the alternative hypothesis,
        must be one of "two.sided" (default), "greater" or "less"


    returns:
    if ((n is None) and (d is not None)):
        result = pwr.pwr_t_test(d=d, sig_level=sig_level,
                                power=power, type=type,
                                alternative=alternative)
        return result
    elif ((n is not None) and (d is None)):
        result = pwr.pwr_t_test(n=n, sig_level=sig_level,
                                power=power, type=type,
                                alternative=alternative)
        return result


    '''
    try:
        pwr = importr("pwr")
    except RRuntimeError as e:
        print("Make sure R library pwr is installled")
        print("Error:", e)

    #   pwr.t.test(
    #     n = rinterface.NULL,
    #     d = rinterface.NULL,
    #     sig_level = 0.05,
    #     power = rinterface.NULL,
    #     type = c,
    #     alternative = c,
    # )
    #       Exactly one of the parameters 'd','n','power' and
    #       'sig.level' must be passed as NULL, and that parameter is
    #       determined from the others. Notice that the last one has non-NULL
    #       default so NULL must be explicitly passed if you want to compute
    #       it.

    if ((n is None) and (d is not None)):
        result = pwr.pwr_t_test(d=d, sig_level=sig_level,
                                power=power, type=type,
                                alternative=alternative)
        return result

    elif ((n is not None) and (d is None)):
        result = pwr.pwr_t_test(n=n, sig_level=sig_level,
                                power=power, type=type,
                                alternative=alternative)
        return result
