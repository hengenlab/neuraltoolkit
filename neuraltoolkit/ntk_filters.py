#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All filter, power  functions

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1

List of functions/class in ntk_filters
butter_bandpass(data, highpass, lowpass, fs, order=3)
butter_lowpass(data, lowpass, fs, order=3)
butter_highpass(data, highpass, fs, order=3)
welch_power(data, fs, lengthseg, noverlappoints, axis_n=-1, lplot=0)
notch_filter(data, fs, Q, ftofilter)
'''


# Butterworth filters
def butter_bandpass(data, highpass, lowpass, fs, order=3):

    '''
    Butterworth bandpass filter
    butter_bandpass(data, highpass, lowpass, fs, order=3)
    result = butter_bandpass(data, 500, 4000, 25000, 3)
    '''

    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    high_pass = highpass / nyq
    low_pass = lowpass / nyq
    b, a = butter(order, [high_pass, low_pass], btype='bandpass')
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(data, lowpass, fs, order=3):

    '''
    Butterworth lowpass filter
    butter_lowpass(data, lowpass, fs, order=3)
    result = butter_lowpass(data, 500,  25000, 3)
    '''

    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low_pass = lowpass / nyq
    b, a = butter(order, [low_pass], btype='lowpass')
    y = filtfilt(b, a, data)
    return y


def cheby1_lowpass(data, lowpass, fs, order=3, ripple=0.8):

    '''
    Chebyshev type I lowpass filter
    cheby1_lowpass(data, lowpass, fs, order=3, ripple=0.8)
    result = cheby1_lowpass(data, 500,  25000, order=3, ripple=0.8)
    '''

    from scipy.signal import cheby1, filtfilt
    nyq = 0.5 * fs
    low_pass = lowpass / nyq
    b, a = cheby1(order, ripple, low_pass, 'low')
    y = filtfilt(b, a, data)
    return y


def butter_highpass(data, highpass, fs, order=3):

    '''
    Butterworth highpass filter
    butter_bandpass(data, lowpass, fs, order=3)
    result = butter_highpass(data, 500,  25000, 3)
    '''

    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    high_pass = highpass / nyq
    b, a = butter(order, [high_pass], btype='highpass')
    y = filtfilt(b, a, data)
    return y


def welch_power(data, fs, lengthseg, noverlappoints, axis_n=-1, lplot=0):

    '''
    Welch power of signal

    welch_power(data, fs, lengthseg, noverlappoints, axis_n=-1, lplot=0):

    fs sampling frequency
    lengthseg : length of segments
    noverlappoints : number of points to overlap between segments
    axis_n : deault -1

    returns
    sf : sample frequencies
    px : power spectral density

    welch_power(data, 25000, 2048, 0, -1, 0)
    '''

    from scipy.signal import welch
    import matplotlib.pyplot as plt

    sf, px = welch(data, fs, window='hann', nperseg=lengthseg,
                   noverlap=noverlappoints, axis=axis_n)

    if lplot == 1:
        plt.semilogy(sf, px)
        plt.show()

    return sf, px


def notch_filter(data, fs, Q, ftofilter):

    '''
    Notch filter

    notch_filter(data, fs, Q, ftofilter)

    fs : sampling frequency
    Q :  Q = w0/bw, w0 = ftofilter/(fs/2), bw bandwidth
    ftofilter : frequency to filter out

    return
    datan : filtered signal

    '''

    from scipy.signal import filtfilt, iirnotch

    w0 = ftofilter/(fs/2)  # Normalized Frequency
    print(w0)
    print(w0/10)

    # notch filter
    b, a = iirnotch(w0, Q)

    # filter foward and reverse
    datan = filtfilt(b, a, data)

    return datan


def ntk_spectrogram(lfp, fs, nperseg_fact=4, f_low=1, f_high=64,
                    lsavefile=None):

    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import numpy as np
    import os

    '''
    plot spectrogram

    ntk_spectrogram(lfp, fs, nperseg_fact=4, f_low=1, f_high=64,
                    lsavefile=None)

    lfp : lfp one channel
    fs : sampling frequency
    nperseg_fact : multiple of fs to get length of each segment
    f_low : filter frequencies below f_low
    f_high : filter frequencies above f_high
    lsaveloc : default None (show plot), if path is give save fig
               to path. for example
               lsavefile='/home/kbn/spec_1.jpg'

    Example:
    ntk.ntk_spectrogram(lfp_all[0, :], 250, 4, 1, 64,
                        lsavefile='/home/kbn/spec_1.jpg')

    '''

    if lsavefile is not None:
        # Check directory exists
        if not (os.path.exists(os.path.split(lsavefile)[0]) and
                os.path.isdir(os.path.split(lsavefile)[0])):
            raise NotADirectoryError("Directory {} does not exists".
                                     format(os.path.split(lsavefile)[0]))

    nperseg = fs * nperseg_fact
    noverlap = fs * nperseg_fact * 0.5
    # nperseg length of each segment : (1sec = fs)
    # nfft  same as nperseg
    # noverlap = 0.5 sec  = 1/2fs
    f, t_spec, x_spec = signal.spectrogram(lfp, fs=fs, window='hanning',
                                           nperseg=nperseg,
                                           noverlap=noverlap,
                                           detrend=False,  mode='psd')
    # Remove noise
    x_mesh, y_mesh = np.meshgrid(t_spec, f[(f < f_high) & (f > f_low)])
    plt.figure(figsize=(16, 2))
    plt.pcolormesh(x_mesh, y_mesh,
                   np.log10(x_spec[(f < f_high) & (f > f_low)]),
                   cmap='jet')
    plt.yscale('log')
    # Add this if Sleep Wake scoring GUI has space
    # plt.xlabel('Time in minutes')
    # plt.ylabel('Frequency (log)')
    # plt.tight_layout()
    if lsavefile is None:
        plt.show()
    else:
        plt.savefig(lsavefile)
