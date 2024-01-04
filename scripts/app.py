# flake8: noqa
import neuraltoolkit as ntk
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


st.title('Plot ecube data')

def file_selector(folder_path='/media/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

rawfile = file_selector()

number_of_channels = st.sidebar.slider("Total number of channels", 1, 640)
nprobes = st.sidebar.slider("Select a Number of probes", 1, 10)
probenum = st.sidebar.slider("Select probe Number", 1, 10) -1
samples = st.sidebar.slider("Number of seconds to plot", 1, 300) * 25000

selected_options = st.multiselect("Select one or more options:",
    ['A', 'B', 'C'])

channel_num = st.sidebar.slider("Select channel number to plot", 1, 64) -1




# Load only one probe from raw file
# rawfile = '/media/bs007r/CAF00099/CAF00099_L9_W2_2021-06-07_11-05-32/Headstages_512_Channels_int16_2021-06-07_21-15-33.bin'
# number_of_channels = 512
fs = 25000
hstype = ['APT_PCB', 'APT_PCB', 'APT_PCB',
          'APT_PCB', 'APT_PCB', 'APT_PCB',
          'APT_PCB', 'APT_PCB']   # Channel map
# ts = 0, start from begining of file or can be any sample number
# te = 2500, read 2500 sample points from ts ( te greater than ts)
# if ts =0 and te = -1,  read from begining to end of file
# nprobes = 8
# probenum = 0  # which probe to return (starts from zero)
probechans = 64  #  number of channels per probe (symmetric)

@st.cache
def load(rawfile,   number_of_channels,   hstype,   nprobes,  samples,     probenum):
    print(rawfile,   number_of_channels,   hstype,   nprobes, samples,      probenum)
    t,dgc = ntk.load_raw_gain_chmap_1probe(rawfile, number_of_channels,
                                           hstype, nprobes=nprobes,
                                           lraw=1, ts=0, te=samples,
                                           probenum=probenum, probechans=64)

    # bandpass filter
    bdgc = ntk.butter_bandpass(dgc, 500, 7500, fs, 3)
    return dgc, bdgc



if st.button('Run'):
    st.write("Loading rawfile")
    raw, braw = load(rawfile,   number_of_channels,   hstype,   nprobes, samples,       probenum)
    st.write("Done loading data press run if any loading parameter is changed")
    st.write("To plot different channels select channel number to plot and press plot")

if st.button('Plot'):
    raw, braw = load(rawfile,   number_of_channels,   hstype,   nprobes, samples,      probenum)
    st.write("Plotting")
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(raw[channel_num, :])
    ax[1].plot(braw[channel_num, :])
    st.pyplot(fig)

