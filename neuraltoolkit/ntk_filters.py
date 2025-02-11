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
    """
    Apply a notch filter to remove a specific frequency from the signal.

    Parameters:
    -----------
    data : array_like
        The input signal to be filtered.
    fs : float
        The sampling frequency of the input signal in Hz.
    Q : float
        The quality factor of the notch filter, defined as Q = f0 / bw,
        where f0 is the frequency to remove (ftofilter) and
        bw is the bandwidth.
        - High Q (e.g., 30): Provides a narrow notch filter, removing a very
          specific frequency with minimal impact on adjacent frequencies.
          Suitable for isolating and removing narrowband interference.
        - Low Q (e.g., 5): Creates a wider notch, affecting a broader range
          of frequencies, but may remove more nearby frequencies.
    ftofilter : float
        The frequency (in Hz) to be filtered out from the signal. For example,
        set `ftofilter = 60` to remove 60 Hz power line noise.

    Returns:
    --------
    datan : array_like
        The filtered signal with the specified frequency removed.

    Notes:
    ------
    The notch filter is implemented using a forward-backward filter
    (via `filtfilt`), which ensures zero phase distortion in
    the filtered signal.

    Example:
    --------
    # Example usage:
    filtered_data = ntk.notch_filter(data, fs=25000, Q=30, ftofilter=60)

    This example applies a notch filter to remove 60 Hz noise
    from a signal sampled at 25000 Hz.
    """

    from scipy.signal import filtfilt, iirnotch

    # Normalized frequency: f0 / (fs / 2)
    w0 = ftofilter / (fs / 2)

    # Design the notch filter
    b, a = iirnotch(w0, Q)

    # Apply the notch filter in both forward and reverse directions
    datan = filtfilt(b, a, data)

    return datan


def ntk_spectrogram(lfp, fs, nperseg=None, noverlap=None, f_low=1, f_high=64,
                    lsavedir=None, hour=0, chan=0, reclen=3600,
                    lsavedeltathetha=0,
                    probenum=None,
                    lmultitaper=0):

    '''
    plot spectrogram and save delta and thetha

    ntk_spectrogram(lfp, fs, nperseg, noverlap, f_low=1, f_high=64,
                    lsavedir=None, hour=0, chan=0, reclen=3600,
                    lsavedeltathetha=0,
                    probenum=None,lmultitaper=0)

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
    lmultitaper : use multitaper spectrogram

    Example:
    ntk.ntk_spectrogram(lfp_all[0, :], fs, nperseg, noverlap, 1, 64,
                        lsavedir='/home/kbn/',
                        hour=0, chan=0, reclen=3600, lsavedeltathetha=0,
                        probenum=None,
                        lmultitaper=1)

    '''

    # Add a small value, EPS,
    # helps to avoid numerical instability (e.g., division by zero) when
    # lfp has near-zero values and mitigates precision errors during
    # spectrogram computations. This ensures stable and accurate results.
    EPS = 1e-10

    # window_length and step size
    window_length = 4
    step_size = 2

    if lsavedir is not None:
        # Check directory exists
        if not (os.path.exists(lsavedir) and
                os.path.isdir(lsavedir)):
            raise NotADirectoryError("Directory {} does not exists".
                                     format(lsavedir))

    if lmultitaper:
        # from mtspec import multitaper_spectrogram
        import multitaper_toolbox as mt
        f0, t_spec0, x_spec0, (fig, ax) = \
            mt.multitaper_spectrogram(
            data=lfp + EPS,
            fs=fs,
            # 4-second windows, 2-second step
            window_params=[window_length, step_size],
            # Limit frequencies from 0 to 32 Hz
            frequency_range=[0, 32],
            multiprocess=False,
            # Number of tapers
            num_tapers=5,
            #  Number of Tapers = 2 × time-bandwidth product (NW) − 1
            time_bandwidth=3,
            # 'constant', 'linear',
            detrend_opt='linear',
            plot_on=True,
            return_fig=True,
        )

    if nperseg is None:
        nperseg = fs * window_length
    if noverlap is None:
        noverlap = fs * step_size
    try:
        f, t_spec, x_spec = signal.spectrogram(lfp + EPS, fs=fs,
                                               window='hann',
                                               nperseg=nperseg,
                                               noverlap=noverlap,
                                               detrend=False,  mode='psd')
    except Exception as e:
        print("Error ", e, " changing window to hann")
        f, t_spec, x_spec = signal.spectrogram(lfp + EPS, fs=fs,
                                               window='hanning',
                                               nperseg=nperseg,
                                               noverlap=noverlap,
                                               detrend=False,  mode='psd')
    # print("sh x_spec ", x_spec.shape)
    # Remove noise
    if not lmultitaper:
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
        # # Extract delta, thetha
        # delt = sum(x_spec[np.where(np.logical_and(f >= 1, f <= 4))])
        # thetw = sum(x_spec[np.where(np.logical_and(f >= 2, f <= 16))])
        # thetn = sum(x_spec[np.where(np.logical_and(f >= 5, f <= 10))])
        if f.ndim == 1:
            # Without multitaper case
            delta_band = np.logical_and(f >= 0, f <= 4)
            delt = np.sum(x_spec[delta_band, :], axis=0)

            theta_wide_band = np.logical_and(f >= 2, f <= 16)
            thetw = np.sum(x_spec[theta_wide_band, :], axis=0)

            theta_narrow_band = np.logical_and(f >= 5, f <= 10)
            thetn = np.sum(x_spec[theta_narrow_band, :], axis=0)

        else:
            # With multitaper case
            delta_band = np.logical_and(f[:, 0] >= 0, f[:, 0] <= 4)
            delt = np.sum(x_spec[delta_band]) * np.ones(f.shape[1])

            theta_wide_band = np.logical_and(f[:, 0] >= 2, f[:, 0] <= 16)
            thetw = np.sum(x_spec[theta_wide_band]) * np.ones(f.shape[1])

            theta_narrow_band = np.logical_and(f[:, 0] >= 5, f[:, 0] <= 10)
            thetn = np.sum(x_spec[theta_narrow_band]) * np.ones(f.shape[1])
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
            if chan is None:
                np.save(os.path.join(lsavedir, 'delt' +
                                     str(hour) + '.npy'),
                        delt)
                np.save(os.path.join(lsavedir, 'thet' +
                                     str(hour) + '.npy'),
                        thet)
            else:
                np.save(os.path.join(lsavedir, 'delt' + str(hour) +
                                     '_ch' + str(chan) +
                                     + '.npy'), delt)
                np.save(os.path.join(lsavedir, 'thet' + str(hour) +
                                     '_ch' + str(chan) +
                                     + '.npy'), thet)

        else:
            if chan is None:
                np.save(os.path.join(lsavedir, 'delt' + str(hour) +
                                     '_probe' + str(probenum) + '.npy'),
                        delt)
                np.save(os.path.join(lsavedir, 'thet' + str(hour) +
                                     '_probe' + str(probenum) + '.npy'),
                        thet)
            else:
                np.save(os.path.join(lsavedir, 'delt' + str(hour) +
                                     '_ch' + str(chan) +
                                     '_probe' + str(probenum) + '.npy'),
                        delt)
                np.save(os.path.join(lsavedir, 'thet' + str(hour) +
                                     '_ch' + str(chan) +
                                     '_probe' + str(probenum) + '.npy'),
                        thet)

    plt.close('all')


def selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=64,
                   probenum=0, probechans=64, lfp_lowpass=250,
                   lsavedeltathetha=0,
                   lmultitaper=0):
    '''
    selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=64,
                   probenum=0, probechans=64, lfp_lowpass=250,
                   lsavedeltathetha=0,
                   lmultitaper=0)
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
    lsavedeltathetha : whether to save delta and thetha too default(0)
    lmultitaper : use multitaper spectrogram


    selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=1, number_of_channels=128,
                   probenum=1, probechans=64, lfp_lowpass=250,
                   lsavedeltathetha=0,
                   lmultitaper=0)
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
                        lsavedeltathetha=lsavedeltathetha,
                        lmultitaper=lmultitaper,
                        probenum=probenum)

    print("Finished saving spectrogram for all channels")
