# -*- coding: utf-8 -*-

'''
All filter, power  functions
Hlab
Author :Kiran Bhaskaran-Nair

from neuraltoolkit import ntk_filters as ntkf
help(ntkf)

List of functions in ntk_filters
* butter_bandpass
* butter_highpass
* butter_lowpass

bandpass filter
help(ntkf.butter_bandpass)

lowpass filter
help(ntkf.butter_lowpass)

highpass filter
help(ntkf.butter_highpass)
'''


# Butterworth filters
def butter_bandpass(data, highpass, lowpass, fs, order=3):

   '''
   Butterworth bandpass filter
   butter_bandpass(data, highpass, lowpass, fs, order=3)
   result = butter_bandpass(data, 500, 4000, 25000, 3) 
   '''

   from scipy.signal import butter, lfilter, filtfilt

   nyq = 0.5 * fs
   high_pass = highpass / nyq
   low_pass  = lowpass / nyq
   b, a = butter(order, [high_pass, low_pass], btype='bandpass')
   y = filtfilt(b, a, data)
   return y

def butter_lowpass(data, lowpass, fs, order=3):

   '''
   Butterworth lowpass filter
   butter_lowpass(data, lowpass, fs, order=3)
   result = butter_lowpass(data, 500,  25000, 3) 
   '''

   from scipy.signal import butter, lfilter, filtfilt

   nyq = 0.5 * fs
   low_pass  = lowpass / nyq
   b, a = butter(order, [low_pass], btype='lowpass')
   y = filtfilt(b, a, data)
   return y

def butter_highpass(data, highpass, fs, order=3):

   '''
   Butterworth highpass filter
   butter_bandpass(data, lowpass, fs, order=3)
   result = butter_highpass(data, 500,  25000, 3) 
   '''

   from scipy.signal import butter, lfilter, filtfilt

   nyq = 0.5 * fs
   high_pass = highpass / nyq
   b, a = butter(order, [high_pass], btype='highpass')
   y = filtfilt(b, a, data)
   return y



def welch_power(data, fs, lengthseg, noverlappoints, axis_n=-1, lplot=0):
    from scipy.signal import welch
    import matplotlib.pyplot as plt

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
    
    sf, px = welch(data, fs, window='hann', nperseg=lengthseg, noverlap=noverlappoints, axis=axis_n )

    if lplot == 1:
        plt.semilogy(sf, px)
        plt.show()
        
    return sf, px



def notch_filter(data, fs, Q, ftofilter):
    from scipy.signal import filtfilt, iirnotch

    '''
    Notch filter

    notch_filter(data, fs, Q, ftofilter)

    fs : sampling frequency
    Q :  Q = w0/bw, w0 = ftofilter/(fs/2), bw bandwidth
    ftofilter : frequency to filter out

    return
    datan : filtered signal

    '''

    w0 = ftofilter/(fs/2)  # Normalized Frequency
    print(w0)
    print(w0/10)

    # notch filter
    b, a = iirnotch(w0, Q)

    # filter foward and reverse
    datan = filtfilt(b, a, data)

    return datan
