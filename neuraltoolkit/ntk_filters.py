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
import numpy as np
import os
import glob
import scipy.signal as signal
from scipy.signal import butter, filtfilt
from scipy.signal import cheby1
import matplotlib.pyplot as plt
from neuraltoolkit import ntk_ecube


# Butterworth filters
def butter_bandpass(data, highpass, lowpass, fs, order=3):

    '''
    Butterworth bandpass filter
    butter_bandpass(data, highpass, lowpass, fs, order=3)
    result = butter_bandpass(data, 500, 4000, 25000, 3)
    '''

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
                    lsavedir=None, hour=0, chan=0, reclen=3600,
                    lsavedeltathetha=0,
                    probenum=None):

    '''
    plot spectrogram and save delta and thetha

    ntk_spectrogram(lfp, fs, nperseg, noverlap, f_low=1, f_high=64,
                    lsavedir=None, hour=0, chan=0, reclen=3600,
                    lsavedeltathetha=0,
                    probenum=None)

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
    chan: by default 0
    reclen: one hour in seconds (default 3600)
    lsavedeltathetha : whether to save delta and thetha too
    probenum : which probe to return (starts from zero)

    Example:
    ntk.ntk_spectrogram(lfp_all[0, :], fs, nperseg, noverlap, 1, 64,
                        lsavedir='/home/kbn/',
                        hour=0, chan=0, reclen=3600, lsavedeltathetha=0,
                        probenum=None)

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
    # print("sh x_spec ", x_spec.shape)
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
        if probenum is None:
            if chan is None:
                plt.savefig(os.path.join(lsavedir, 'specthr' +
                                         str(hour) + '.jpg'))
            else:
                plt.savefig(os.path.join(lsavedir, 'specthr' +
                                         '_ch' + str(chan) +
                                         str(hour) + '.jpg'))
        else:
            if chan is None:
                plt.savefig(os.path.join(lsavedir, 'specthr' + str(hour) +
                                         '_probe' + str(probenum) + '.jpg'))
            else:
                plt.savefig(os.path.join(lsavedir, 'specthr' + str(hour) +
                                         '_ch' + str(chan) +
                                         '_probe' + str(probenum) + '.jpg'))

    if lsavedeltathetha:
        # Extract delta, thetha
        delt = sum(x_spec[np.where(np.logical_and(f >= 1, f <= 4))])
        thetw = sum(x_spec[np.where(np.logical_and(f >= 2, f <= 16))])
        thetn = sum(x_spec[np.where(np.logical_and(f >= 5, f <= 10))])
        thet = thetn/thetw

        # Normalize delta and thetha
        delt = (delt-np.average(delt))/np.std(delt)
        thet = (thet-np.average(thet))/np.std(thet)
        # print("sh delt ", delt.shape, " sh thet ", thet.shape)

        # Add padding to make it an hour
        dispt = reclen//2 - np.size(thet)
        dispd = reclen//2 - np.size(thet)
        # print("dispt ", dispt, " dispd ", dispd)
        if dispt > 0:
            if dispt > 1:
                print("Added padding")
            thet = np.pad(thet, (0, dispt), 'constant')
        if dispd > 0:
            if dispd > 1:
                print("Added padding")
            delt = np.pad(delt, (0, dispd), 'constant')

        if probenum is None:
            np.save(os.path.join(lsavedir, 'delt' + str(hour) + '.npy'), delt)
            np.save(os.path.join(lsavedir, 'thet' + str(hour) + '.npy'), thet)
        else:
            np.save(os.path.join(lsavedir, 'delt' + str(hour) +
                                 '_probe' + str(probenum) + '.npy'), delt)
            np.save(os.path.join(lsavedir, 'thet' + str(hour) +
                                 '_probe' + str(probenum) + '.npy'), thet)
    plt.close('all')


def selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=64,
                   probenum=0, probechans=64, lfp_lowpass=250):
    '''
    selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=64,
                   probenum=0, probechans=64, lfp_lowpass=250)
    rawdat_dir : raw data directory
    outdir : output dir.
       Standard /media/HlabShare/Sleep_Scoring/ABC00001/LFP_chancheck/'
           : Change ABC00001 to animal name
    hstype : Headstage type,
            ['EAB50chmap_00', 'EAB50chmap_00', 'EAB50chmap_00',
             'EAB50chmap_00', 'EAB50chmap_00', 'EAB50chmap_00',
             'EAB50chmap_00', 'EAB50chmap_00']
    hour: hour to generate spectrograms
    fs : sampling frequency (default 25000)
    nprobes : Number of probes (default 1)
    number_of_channels : total number of channels
    probenum : which probe to return (starts from zero)
    probechans : number of channels per probe (symmetric)
    lfp_lowpass : default 250


    selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=128,
                   probenum=1, probechans=64, lfp_lowpass=250)
    '''
    # from neuraltoolkit import ntk_ecube

    # Check directory exists
    if not (os.path.exists(rawdat_dir) and
            os.path.isdir(rawdat_dir)):
        raise NotADirectoryError("Directory {} does not exists".
                                 format(rawdat_dir))
    if not (os.path.exists(outdir) and
            os.path.isdir(outdir)):
        raise NotADirectoryError("Directory {} does not exists".
                                 format(outdir))

    os.chdir(rawdat_dir)
    files = np.sort(glob.glob('H*.bin'))

    # Select files by hour
    fil = hour*12
    load_files = files[fil:fil+12]

    dat_full = None
    for indx in range(len(load_files)):
        t, dat = ntk_ecube.load_raw_gain_chmap_1probe(load_files[indx],
                                                      number_of_channels,
                                                      hstype,
                                                      nprobes=nprobes,
                                                      lraw=1, te=-1,
                                                      probenum=probenum,
                                                      probechans=probechans)
        if indx == 0:
            dat_full = dat * 1
        else:
            dat_full = np.concatenate((dat_full, dat), axis=1)
    dat = None
    del dat
    print("Finished loading all files")

    # if lswscoring:
    #     lfp_data = butter_lowpass(dat_full[0:5, :], lfp_lowpass, fs, 3)
    #     print("sh lfp_data ", lfp_data.shape)
    #     lfp_data = np.mean(lfp_data, axis=0)
    #     print("sh2 lfp_data ", lfp_data.shape)
    #     ds_rate = int(fs/(2*lfp_lowpass))
    #     lfp_data = np.int16(lfp_data[0::ds_rate])
    #     ntk_spectrogram(lfp_data, int(lfp_lowpass*2), nperseg=None,
    #                     noverlap=None, f_low=1, f_high=64,
    #                     lsavedir=outdir, hour=hour, chan=None, reclen=3600,
    #                     lsavedeltathetha=0,
    #                     probenum=None)
    for chan in np.arange(0, probechans, 1):
        lfp_data = butter_lowpass(dat_full[chan, :], lfp_lowpass, fs, 3)
        ds_rate = int(fs/(2*lfp_lowpass))
        lfp_data = np.int16(lfp_data[0::ds_rate])
        ntk_spectrogram(lfp_data, int(lfp_lowpass*2), nperseg=None,
                        noverlap=None, f_low=1, f_high=64,
                        lsavedir=outdir, hour=hour, chan=chan, reclen=3600,
                        lsavedeltathetha=0,
                        probenum=probenum)

    print("Finished saving spectrogram for all channels")
