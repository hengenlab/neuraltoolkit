# -*- coding: utf-8 -*-

'''
All filter functions
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

