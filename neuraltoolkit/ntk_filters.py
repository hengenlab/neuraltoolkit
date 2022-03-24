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


def ntk_spectrogram(lfp, fs, nperseg=None, noverlap=None, f_low=1, f_high=64,
                    lsavedir=None, hour=0, reclen=3600, lsavedeltathetha=0):

    import matplotlib.pyplot as plt
    import scipy.signal as signal
    import numpy as np
    import os

    '''
    plot spectrogram and save delta and thetha

    ntk_spectrogram(lfp, fs, nperseg, noverlap, f_low=1, f_high=64,
                    lsavedir=None)

    lfp : lfp one channel
    fs : sampling frequency
    nperseg : length of each segment (default fs *4)
    noverlap : number of points to overlap between segments (default fs*2)
    f_low : filter frequencies below f_low
    f_high : filter frequencies above f_high
    lsaveloc : default None (show plot), if path is give save fig
               to path. for example
               lsavedir='/home/kbn/'
    hour: by default 0
    reclen: one hour in seconds (default 3600)

    Example:
    ntk.ntk_spectrogram(lfp_all[0, :], fs, nperseg, noverlap, 1, 64,
                        lsavedir='/home/kbn/')

    '''

    if lsavedir is not None:
        # Check directory exists
        if not (os.path.exists(lsavedir) and
                os.path.isdir(lsavedir)):
            raise NotADirectoryError("Directory {} does not exists".
                                     format(lsavedir))
    if nperseg is None:
        nperseg = fs * 4
    if noverlap is None:
        noverlap = fs * 2

    f, t_spec, x_spec = signal.spectrogram(lfp, fs=fs, window='hanning',
                                           nperseg=nperseg,
                                           noverlap=noverlap,
                                           detrend=False,  mode='psd')
    print("sh x_spec ", x_spec.shape)
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
    if lsavedir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(lsavedir, 'specthr' + str(hour) + '.jpg'))

    if lsavedeltathetha:
        # Extract delta, thetha
        delt = sum(x_spec[np.where(np.logical_and(f >= 1, f <= 4))])
        thetw = sum(x_spec[np.where(np.logical_and(f >= 2, f <= 16))])
        thetn = sum(x_spec[np.where(np.logical_and(f >= 5, f <= 10))])
        thet = thetn/thetw

        # Normalize delta and thetha
        delt = (delt-np.average(delt))/np.std(delt)
        thet = (thet-np.average(thet))/np.std(thet)
        print("sh delt ", delt.shape, " sh thet ", thet.shape)

        # Add padding to make it an hour
        dispt = reclen//2 - np.size(thet)
        dispd = reclen//2 - np.size(thet)
        print("dispt ", dispt, " dispd ", dispd)
        if dispt > 0:
            if dispt > 1:
                print("Added padding")
            thet = np.pad(thet, (0, dispt), 'constant')
        if dispd > 0:
            if dispd > 1:
                print("Added padding")
            delt = np.pad(delt, (0, dispd), 'constant')

        np.save(os.path.join(lsavedir, 'delt' + str(hour) + '.npy'), delt)
        np.save(os.path.join(lsavedir, 'thet' + str(hour) + '.npy'), thet)
