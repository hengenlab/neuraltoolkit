# Neuraltoolkit
---
A powerful and fast set of tools for loading data, filtering, processing,
working with data formats, and basic utilities for electrophysiology and
behavioral data.

---
![Tests](https://github.com/hengenlab/neuraltoolkit/actions/workflows/pytests.yml/badge.svg)

## Installation

### Download neuraltoolkit
```
git clone https://github.com/hengenlab/neuraltoolkit.git 
```

### Using pip
```
cd locationofneuraltoolkit/neuraltoolkit/
For example /home/kbn/git/neuraltoolkit  
pip install .
```
<!--
---
### Installation by adding to path (not recommended)

#### Windows
My Computer > Properties > Advanced System Settings > Environment Variables >  
In system variables, create a new variable  
    Variable name  : PYTHONPATH  
    Variable value : location where neuraltoolkit is located  
    Click OK


#### Linux
If you are using bash shell  
In terminal open .barshrc or .bash_profile  
add this line  
export PYTHONPATH=/location_of_neuraltoolkit:$PYTHONPATH


#### Mac
If you are using bash shell  
In terminal cd ~/  
then open  .profile using your favourite text editor (open -a TextEdit .profile)
to add location where neuraltoolkit is located add the line below

export PYTHONPATH=/location_of_neuraltoolkit:$PYTHONPATH
-->
---

### Test import
```
Open powershell/terminal
    ipython
    import neuraltoolkit as ntk
```

---


## load ecube data


```
import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt

# Load only one probe from raw file
rawfile = '/home/kbn/Headstages_512_Channels_int16_2019-06-21_03-55-09.bin'
number_of_channels = 512
hstype = ['APT_PCB', 'APT_PCB', 'APT_PCB',
          'APT_PCB', 'APT_PCB', 'APT_PCB',
          'APT_PCB', 'APT_PCB']   # Channel map
# ts = 0, start from begining of file or can be any sample number
# te = 2500, read 2500 sample points from ts ( te greater than ts)
# if ts =0 and te = -1,  read from begining to end of file
nprobes = 8
probenum = 0  # which probe to return (starts from zero)
probechans = 64  #  number of channels per probe (symmetric)
t,dgc = ntk.load_raw_gain_chmap_1probe(rawfile, number_of_channels,
                                       hstype, nprobes=nprobes,
                                       lraw=1, ts=0, te=-1,
                                       probenum=probenum, probechans=64)

# bandpass filter
bdgc = ntk.butter_bandpass(dgc, 500, 7500, 25000, 3)

# Time only
t = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, 'hs64', t_only=1)

# Load digital data for cameras, etc
digitalrawfile = '/home/kbn/Digital_64_Channels_int64_2018-11-04_11-18-12.bin'
t_only  : if t_only=1, just return  timestamp
          (Default 0, returns timestamp and data)
lcheckdigi64 : Default 1, check for digital file with 64 channel
          (atypical recordings) and correct values of -11 to 0 and -9 to 1
          lcheckdigi64=0, no checks are done, just read the file and
          returns timestamp and data
tdig, ddig = ntk.load_digital_binary(digitalrawfile, t_only=0, lcheckdigi64=1)

# Load time only from digital data for cameras, etc
tdig = ntk.load_digital_binary(digitalrawfile, t_only=1)


# Load all digital channels new api digital files outputs
import neuraltoolkit as ntk
import matplotlib.pyplot as plt
name = 'DigitalPanel_2_Channels_bool_masked_uint64_2021-04-30_08-46-15.bin'
t, d = ntk.load_digital_binary_allchannels(name, t_only=0, channel=-1)
# d contains all 64 channels
print("sh d ", d.shape)
# plot second channel
plt.plot(d[1, 0:10000])
plt.show()
# Load just one channel. Please remember channel number here starts from 0.
import neuraltoolkit as ntk
import matplotlib.pyplot as plt
name = 'DigitalPanel_2_Channels_bool_masked_uint64_2021-04-30_08-46-15.bin'
t, d = ntk.load_digital_binary_allchannels(name, t_only=0, channel=1)
# d contains only second digital channel
print("sh d ", d.shape)
plt.plot(d[0:10000])
plt.show()

# Get digital event sample times from recording block/session
Get digital event sample times from recording block/session
can be also used to analyze other digital data too,
except watchtower

get_digital_event_sample_times(fl_list, channel_number, val=0, verbose=0)
fl_list : file list of DigitalPanel_*.bin, same as sorting block
          This works only for new digital binary files
channel_number : channel number used to record digital event data,
                 (starts from zero)
val : value to check, in case if digital event is 0 (default)
verbose: to print logs, default (off)

returns:
digital_event_sample_times : sample number when digital event data was 0
t_estart : Ecube time for first digital file
nsamples : total number of samples in all DigitalPanel_*.bin files
           in fl_list

For example:
import glob
import numpy as np
import neuraltoolkit as ntk

fn_dir = '/home/kbn/ABC1234/ABC1234_2021/'
fl_list = np.sort(glob.glob(fn_dir + 'DigitalPanel_*.bin'))
# for second block of sorting (12 hour blocks, each)
fl_list = fl_list[144:288]
channel_number = 2
value = 0
verbose = 0

digital_event_sample_times, t_estart, nsamples = \
ntk.get_digital_event_sample_times(fl_list, channel_number,
                                   val=value, verbose=verbose)

# Light dark transition
datadir = '/media/data/D1/d1_c1/'
l7ampm = 0 # if 1 just check files around 7:00 am and 7:00 pm
lplot = 0
ldt = ntk.light_dark_transition(datadir, l7ampm=0, lplot=0)
ldt - list contains
      [filename, index of light-dark transition in file, start time of the file]

# For example
[['Digital_1_Channels_int64_10.bin', 2743961, 24082251921475],
 ['Digital_1_Channels_int64_13.bin', 2677067, 67284509201475]]

# Visual grating transition
datadir = '/media/bs003r/D1/d1_vg1/'
transition_list = ntk.visual_grating_transition(datadir)
transition_list - list contains
      [filename, indices of visual grating transition in file, time in file]
#For example
Filename Digital_1_Channels_int64_1.bin
index  [  73042  273202  473699  674109  874218 1074640 1275357 1476104 1676162
 7287946 7488659]  time  3783012466437
Filename Digital_1_Channels_int64_2.bin
index  [ 189390  390242  590800  791281  991778 1192327 1392627 1593098 1793569
6005899 6206417 6406882 6607754 6808203 7008951 7209624]  time  4083006706437
Filename Digital_1_Channels_int64_3.bin
index  [5573869 5774268 5974585 6175289 6375922 6576758 6777207 6977770 7177962
7378361]  time  4983018546437


# Find number of samples per channel in a ecube file
rawfile = '/home/kbn/Headstages_512_Channels_int16_2019-06-28_18-13-24.bin'
number_of_channels = 512
lraw is 1 for raw file and for prepocessed file lraw is 0
number_of_samples_per_chan = ntk.find_samples_per_chan(rawfile, 512, lraw=1)

# Find number of samples between two ecube raw or digital files
Assumes there is no corrupt/failed recording files.
binfile1 = '/home/kbn/Headstages_512_Channels_int16_2019-06-28_18-13-24.bin'
binfile2 = '/home/kbn/Digital_1_Channels_int64_2019-06-30_13-11-07.bin'
number_of_channels = 512  # in the raw file
lb1 : default 1, binfile1 is rawfile, 0 if digital file
lb2 : default 1, binfile2 is rawfile, 0 if digital file
samples_between = ntk.samples_between_two_binfiles(binfile1, binfile2, number_of_channels,
                                                   hstype, nprobes=8, lb1=1, lb2=0)

# Create channel mapping file for Open Ephys
import neuraltoolkit as ntk
ntk.create_chanmap_file_for_oe()
Enter total number of probes:
1
Enter total number of channels :
64
Enter probe type :
hs64
Enter filename to save data:
channelmap_hs64.txt


# make_binaryfiles_ecubeformat
import numpy as np
import neuraltoolkit as ntk
filename = '/home/kbn/HH.bin'
ltype = 2 # digital files
t = np.uint64(101)
if ltype == 1:
    data_low = -32000
    data_high = 32000
    data_rows = 64
    data_length = 25000*60*5
    data_type = 'int16'
elif ltype == 2:
    data_low = 0
    data_high = 2
    data_rows = 1
    data_length = 25000*60*5
    data_type = 'int64'
d = np.random.randint(data_low, data_high, (data_rows, data_length),
                      dtype=data_type)
ntk.make_binaryfiles_ecubeformat(t, d, filename, ltype)


# convert ecube_raw_to_preprocessed all channels or just a tetrode
import neuraltoolkit as ntk
rawfile - name of rawfile with path, '/home/ckbn/Headstage.bin'
rawfile = '/home/ckbn/Headstages_64.bin'
outdir - directory to save preprocessed file, '/home/ckbn/output/'
outdir ='/home/ckbn/out/'
number_of_channels - number of channels in rawfile
number_of_channels = 64
hstype : Headstage type, 'hs64' (see manual for full list)
hstype = ['hs64']
nprobes : Number of probes (default 1)
ts = 0, start from begining of file or can be any sample number
te = 2500, read 2500 sample points from ts ( te greater than ts)
if ts=0 and te = -1,  read from begining to end of file
ntk.ecube_raw_to_preprocessed(rawfile, outdir
                              number_of_channels,
                              hstype, nprobes=1,
                              ts=0, te=25000,
                              tetrode_channels=[0,1,2,3])
```
---
## load intan data


```
# import libraries
import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt

# Get filename
rawfile = 'neuraltoolkit/intansimu_170807_205345.rhd'

# Get number of channels
print("Enter total number of channels : ")
number_of_channels = np.int16(eval(input()))

# Time and data
t, dgc = ntk.load_intan_raw_gain_chanmap(rawfile, number_of_channels, 'intan32')

# Time and data for multiple probes with same number of channels
hstype = ['intan32', 'linear']
nprobes = 2
# number_of_channels here is total number of channels in all probes (32 * nprobes = 64)
t, dgc = ntk.load_intan_raw_gain_chanmap(rawfile, number_of_channels, hstype, nprobes)

# Time, data, digital input ( for patch)
t, dgc, din = ntk.load_intan_raw_gain_chanmap(rawfile, number_of_channels, 'intan32', ldin=1)

# bandpass filter
bdgc = ntk.butter_bandpass(dgc, 500, 7500, 25000, 3)

# plot raw data
ntk.plot_data(dgc, 0, 25000, 1)

# plot bandpassed data
ntk.plot_data(bdgc, 0, 25000, 1)

# plot data in channel list
l = np.array([5, 13])
ntk.plot_data_chlist(bdgc, 25000, 50000, l )

# load aux binary data
import neuraltoolkit as ntk
import matplotlib.pyplot as plt
import numpy as np
aux_file = 'Acc_auxtest_191108_102919_t_0#145.8719_l_2917440_p_0_chg_1_aux3p74em5.bin'
auxd = ntk.load_aux_binary_data(aux_file, 3)
x_accel = auxd[0, :]
y_accel = auxd[1, :]
z_accel = auxd[2, :]
# sampling rate for aux is rawdata sampling rate/4
x = np.arange(0, auxd.shape[1]*4, 4)
plt.subplot(3, 1, 1)
plt.plot(x, x_accel, '.-')
plt.title('Plot accelorometer data')
plt.ylabel('X acceleration')
plt.ylim((0.0, 2.5))
plt.subplot(3, 1, 2)
plt.plot(x, y_accel, '.-')
plt.ylabel('Y acceleration')
plt.ylim(0.0, 2.5)
plt.subplot(3, 1, 3)
plt.plot(x, z_accel, '.-')
plt.xlabel('time')
plt.ylabel('Z acceleration')
plt.ylim(0.0, 2.5)
plt.show()
 

```
---
## video

```
import neuraltoolkit as ntk
videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
lstream = 0

# get video attributes
v = ntk.NTKVideos(videofilename, lstream)
print(v.fps)
30.00
print(v.width)
640.0
print(v.height)
480.0
print(v.length)
107998.0

# play video, please press q to exit
v.play_video()

# extract_frames and save to a folder
outpath = '/home/user/out/'
v.extract_frames(outpath)

# Grab a frame and write to disk
frame_num = 1
outpath = '/home/user/out/'
v.grab_frame_num(frame_num, outpath)

# Grab a frame and show
frame_num = 100
v.grab_frame_num(frame_num)

# Play video from framenum
# please press q to exit
# press spacebar to pause
#
# framenum to start from (default 0) starts from begining
# timeinsec (default None), if timeinsec is not None, then
# framenum is calculated based on timeinsec and v.fps
# firstframewaittime (default 5000) first frames waittime
# otherframewaittime (default 10) higher slower video plays
#
v.play_video_from_framenum(framenum=100, timeinsec=None, firstframewaittime=5000, otherframewaittime=10)
#
v.play_video_from_framenum(framenum=100, timeinsec=3.3, firstframewaittime=5000, otherframewaittime=10) 

# Load all video files and return length
import neuraltoolkit as ntk
import glob
import numpy as np
videofile_list =  np.sort(glob.glob('/media/bs001r/watchtower2_tmp/PVCre_animals/KDR00035-*.mp4'))
lstream = 0
v_lengths = None
v_lengths = []
for indx, fl in enumerate(videofile_list):
    print(indx, " ", fl)
    v = ntk.NTKVideos(fl, lstream)
    v_lengths.append([fl, v.length])
for vfl in v_lengths:
    print(vfl)


# Read video files and return list of all video lengths
v_lengths = ntk.get_video_length_list('/home/kbn/watchtower_current/data/')

# Convert video to grey
videofilename = '/media/bs001r/ckbn/opencv/e3v8102-20190711T0056-0156.mp4'
lstream = 0
output_path = '/media/bs001r/ckbn/opencv/'
v = ntk.NTKVideos(videofilename, lstream)
v.grayscale_video(output_path)

# diff video
v.graydiff_video(output_path)
# diff image
v.graydiff_img(output_path)




# Make video from images
imgpath = '/home/user/img/'
videopath = '/home/user/img/out/'
videofilename = video1.avi
ntk.make_video_from_images(imgpath, videopath, videofilename,
                           imgext='.jpg', codec='XVID', v_fps=30)

# Play numpy movie file
import neuraltoolkit as ntk
npmoviefile = '/home/kbn/ns_118images1000fpsnormalized.npy'
ntk.play_numpy_movie(npmoviefile, wait=10, start=500, end=1000,
                     movietitle="naturalimages", verbose=1)

# Extract frames indices in list (framestograb) and write new video file
videofilename = '/home/user/e3v810a-20190307T0740-0840.mp4'
lstream = 0 # as video is already saved
v  = ntk.NTKVideos(videofilename, lstream)
# v contains length, width, height information from video
# for example write after 100 frames grab 10 seconds to new video
framestograb = list(range(100, 100 + int(v.fps*10), 1))
fl_out = None
# if fl_out is not None give full path and name to save video
# fl_out = '/home/kbn/video/video_1.mp4'
# else if fl_out is None new video is saved as
# /home/user/e3v810a-20190307T0740-0840_subblocks.mp4

v.grab_frames_to_video(framestograb=framestograb, fl_out=fl_out)


```

# dlc
```
import neuraltoolkit as ntk
dlc_h5file = 'D17_trial1DeepCut_resnet50_crickethuntJul18shuffle1_15000.h5'
cutoff : cutoff based on confidence
pos, fnames = ntk.dlc_get_position(dlc_h5file, cutoff=0.6)
pos : contains x, y positions for all features
fnames : name of all features
# For example
pos
array([[357.29413831, 439.93870854, 482.14195955, ..., 159.27687836,
        469.79700255, 183.82241535],
       ...,
       [         nan,          nan,          nan, ...,          nan,
                 nan,          nan]])
fnames
['cricket', 'snout', 'tailbase', 'leftear', 'rightear']



find_video_start_index(datadir, ch, nfiles=10,
                       fs=25000, fps=15,
                       lnew=1,
                       fig_indx=None)
# datadir: data directory where digital file is located
# ch : channel where Watchtower signal is recorded,
#      remember number starts from 0
# nfiles: First how many files to check for pulse change
#      (default first 10 files)
# fs: Sampling rate of digital file (default 25000)
# fps: Frames per second of video file (default 15)
# lnew: default 1, new digital files.
#      0 for old digital files with only one channel
# fig_indx: Default None, if index is given it plot figure
datadir = '/home/kbn/ABC12345/ABC_L9_W2_/'
ch = 1   #  _L9_W2_  zero indexing
nfiles = 10
fs = 25000
fps = 15
fig_indx = 1
video_start_index =\
    ntk.find_video_start_index(datadir, ch, nfiles=nfiles,
                                   fs=fs, fps=fps,
                                   lnew=1, fig_indx=fig_indx)
# Please remember video_start_index is continous.
```

---
## filters

```

# import libraries
import neuraltoolkit as ntk
import numpy as np
from matplotlib import pyplot as plt

# load raw data
rawdata = np.load('P_Headstages_64_Channels_int16_2018-11-15_14-30-49.npy')

# bandpass filter
help(ntk.butter_bandpass)
result = ntk.butter_bandpass(rawdata, 500, 4000, 25000, 3)

# Plot result
plt.plot(result[1,0:25000])
plt.show()

# lowpass filter for lfp
fs = 25000
lowpass = 250
lfp = ntk.butter_lowpass(rawdata, lowpass, fs, order=3)

# spectrogram
ntk_spectrogram(lfp, fs, nperseg, noverlap, f_low=1, f_high=64,
                lsavedir=None, hour=0, chan=0, reclen=3600,
                lsavedeltathetha=0,
                probenum=None)

lfp : lfp one channel
fs : sampling frequency
nperseg : length of each segment if None, default fs *4
noverlap : number of points to overlap between segments if None, default fs *2
f_low : filter frequencies below f_low
f_high : filter frequencies above f_high
lsavedir : default None (show plot), if path is give save fig
           to path. for example
           lsavefile='/home/kbn/'
hour: by default 0
chan: by default 0
reclen: one hour in seconds (default 3600)
lsavedeltathetha : whether to save delta and thetha too
probenum : which probe to return (starts from zero)


Example
ntk.ntk_spectrogram(lfp_all[0, :], fs, nperseg, noverlap, 1, 64,
                    lsavefile=None, hour=0, chan=0,
                    reclen=3600, lsavedeltathetha=0,
                    probenum=1)


# Select channels for lfp extraction
import neuraltoolkit as ntk

# Select LFP channels
rawdat_dir='/media/KDR00032/KDR00032_L1_W2_2022-01-24_09-08-46/'
# Standard /media/HlabShare/Sleep_Scoring/ABC00001/LFP_chancheck/'
outdir='/media/HlabShare/Sleep_Scoring/ABC00001/LFP_chancheck/'
hstype = ['APT_PCB', 'APT_PCB']
# hour: hour to generate spectrograms
hour = 0
# fs: sampling frequency (default 25000)
#    nprobes : Number of probes (default 1)
#    number_of_channels : total number of channels
#    probenum : which probe to return (starts from zero)
#    probechans : number of channels per probe (symmetric)
#    lfp_lowpass : default 250


ntk.selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=2, number_of_channels=128,
                   probenum=0, probechans=64, lfp_lowpass=250)
ntk.selectlfpchans(rawdat_dir, outdir, hstype, hour,
                   fs=25000, nprobes=2, number_of_channels=128,
                   probenum=1, probechans=64, lfp_lowpass=250)

```
---
## high dimensional data


```
# TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neuraltoolkit as ntk
data = np.random.rand(800, 4)
# Please adjust parameters, according to data
# This is just an interface
u = ntk.highd_data_tsne(data, perplexity=30.0, n_components=2,
                        metric='euclidean', n_iter=3000,
                        verbose=True)
plt.scatter(u[:,0], u[:,1], c=data)

# UMAP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neuraltoolkit as ntk
data = np.random.rand(800, 4)
# Please adjust parameters, according to data
# This is just an interface
u = ntk.highd_data_umap(data, n_neighbors=40, n_components=2,
                        metric='euclidean', min_dist=0.2,
                        verbose=True)
plt.scatter(u[:,0], u[:,1], c=data)
```

## math

```
# import libraries
import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt

# interpolate
t = np.arange(0, 10)
d = np.cos(t)
plt.figure(1)
plt.plot(t,d)
plt.show(block=False)
tn, d_tn = ntk.data_intpl(t, d, 4, intpl_kind='cubic')
plt.figure(2)
plt.plot(tn,d_tn)
plt.show(block=False)
```
---
## Channel mappings
###### 'hs64'
      [26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,  4,  28, 32, 24, 20, 
      48,  44, 36, 40, 64, 60, 52, 56, 54, 50, 42, 46, 62, 58, 34, 38, 
      39,  35, 59, 63, 47, 43, 51, 55, 53, 49, 57, 61, 37, 33, 41, 45, 
      17,  21, 29, 25, 1,  5 , 13, 9,  11, 15, 23, 19, 3,  7,  31, 27]
 
###### 'eibless-hs64_port32'
      [1,  5,  9,  13, 3,  7,  11, 15, 17, 21, 25, 29, 19, 23, 27, 31, 
      33,  37, 41, 45, 35, 39, 43, 47, 49, 53, 57, 61, 51, 55, 59, 63, 
      2,   6,  10, 14, 4,  8,  12, 16, 18, 22, 26, 30, 20, 24, 28, 32, 
      34,  38, 42, 46, 36, 40, 44, 48, 50, 54, 58, 62, 52, 56, 60, 64]
       
###### 'eibless-hs64_port64'
      [1,  5,  3,  7,  9,  13, 11, 15, 17, 21, 19, 23, 25, 29, 27, 31, 
      33,  37, 35, 39, 41, 45, 43, 47, 49, 53, 51, 55, 57, 61, 59, 63, 
      2,   6,  4,  8,  10, 14, 12, 16, 18, 22, 20, 24, 26, 30, 28, 32, 
      34,  38, 36, 40, 42, 46, 44, 48, 50, 54, 52, 56, 58, 62, 60, 64 ]
       
###### 'intan32'
      [25, 26, 27, 28, 29, 30, 31, 32, 1,  2,  3,  4,  5,  6,  7,  8, 
      24,  23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]
       
###### 'Si_64_KS_chmap'
      [7,  45, 5,  56, 4,  48, 1,  62, 9,  53, 10, 42, 14, 59, 13, 39, 
      18,  49, 16, 36, 23, 44, 19, 33, 26, 40, 22, 30, 31, 35, 25, 27, 
      3,   51, 2,  63, 8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54, 
      21,  55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41, 38, 47, 32, 37]
       
###### 'Si_64_KT_T1_K2_chmap'
      chan_map = ...
      [14, 59, 10, 42, 9,  53, 1,  62, 4,  48, 5,  56, 7,  45, 13, 39,  
      18,  49, 16, 36, 23, 44, 19, 33, 26, 40, 22, 30, 31, 35, 25, 27,  
      3,   51, 2,  63, 8,  64, 6,  61, 12, 60, 11, 57, 17, 58, 15, 54,  
      21,  55, 20, 52, 29, 50, 24, 46, 34, 43, 28, 41, 38, 47, 32, 37]
       
###### 'PCB_tetrode'
        [2, 41, 50, 62, 6, 39, 42, 47, 34, 44, 51, 56,  
        38, 48, 59, 64, 35, 53, 3, 37, 54, 57, 40, 43,  
        45, 61, 46, 49, 36, 33, 52, 55, 15, 5, 58, 60,  
        18, 9, 63, 1, 32, 14, 4, 7, 26, 20, 10, 13, 19, 
        22, 16, 8, 28, 25, 12, 17, 23, 29, 27, 21, 11, 31, 30, 24]

###### 'EAB50chmap_00'
        [2,  4, 20, 35, 3, 19, 22, 32, 5, 15, 26, 31,
         6,  9, 14, 38, 7, 10, 21, 24,  8, 17, 29, 34,
         12, 13, 16, 28, 25, 27, 37, 47, 36, 39, 46,
         64, 40, 48, 51, 54, 42, 45, 52, 58, 43, 56,
         62, 63, 44, 49, 57, 60, 53, 55, 59, 61, 1,
         11, 18, 23, 30, 33, 41, 50]

###### 'UCLA_Si1'
        [47, 43, 35, 50, 37, 55, 40, 58, 41, 60, 44, 64,
         46, 51, 49, 63, 27, 61, 30, 57, 33, 36, 52, 39,
         59, 42, 45, 48, 53, 56, 62, 54, 15, 1, 5, 9, 4,
         7, 10, 14, 13, 20, 16, 19, 11, 22, 6, 25, 2, 18,
         3, 24, 8, 23, 12, 28, 17, 26, 21, 32, 29, 31, 34,
         38]

###### 'APT_PCB'
        [2, 5, 1, 22, 9, 14, 18, 47, 23, 26, 31, 3, 35, 4,
         7, 16, 34, 21, 12, 10, 29, 17, 8, 13, 11, 6, 38,
         19, 24, 20, 15, 25, 37, 32, 28, 27, 52, 46, 41,
         30, 61, 57, 54, 33, 55, 43, 63, 36, 58, 51, 60,
         42, 40, 50, 64, 48, 59, 49, 44, 45, 62, 56, 53,
         39]

       
###### 'linear'
        [1:number_of_channels]
--- 
## sync
Functions to sync data across: raw neural, sync pulse, video files and frames, sleep state labels, and deep lab cut labels.

#### List of functions
* `map_video_to_neural_data` 
This is the primary method of performing synchronization of multiple data types: neural, video, digital (sync pulse),
  manual sleep scored labels, and DLC (deep lab cut) labels. All values (ecube times, sleep scores, and DLC labels)
  are mapped to each video frame. See the 
  [function documentation](https://github.com/hengenlab/neuraltoolkit/blob/fa096a16a974a599a2f4ee45c9564e5a1a6b9336/neuraltoolkit/ntk_sync.py#L32) 
  in `ntk_sync.py` for detailed documentation on the output format.
* `map_videoframes_to_syncpulse`
This is a lower level function which only reads the Camera Sync Pulse digital data files and aggregates the 
  sequences of 000's and 111's into a summarized for. The output includes an entry per each sequence of 000's and 111's in 
  the Camera Sync Pulse data. See the 
  [function documentation](https://github.com/hengenlab/neuraltoolkit/blob/fa096a16a974a599a2f4ee45c9564e5a1a6b9336/neuraltoolkit/ntk_sync.py#L205)
  in `ntk_sync.py` for detailed documentation on the output format.

#### How to sync video when the digital ecube data is incorrect
In recordings prior to CAF99 the ecube time on the digital channel may not have been properly synced to the 
neural data channel, in these cases you can only sync the video using the filename timestamps in the 
neural and video files (this is +/- 1 second accuracy). This guide shows you how to manually compute the time offset and override
`map_video_to_neural_data` to use the manually computed value instead of the value from the SyncPulse files.

- **Step 1:** Compute the difference between neural data start and video start.
  - Neural data start time is found the first neural data file, for example in CAF42 the neural file
    `Headstages_320_Channels_int16_2020-09-14_17-20-37.bin` has a start time of `17:20:37`.
  - Video data start time is found in the first video MP4 file, for example in CAF42 the video file
    `e3v81a6-20200914T172236-182237.mp4` has a start time of `17:22:36`.
    - Gotchas: Some video file timestamps have an erroneous start time, for example with CAF26 
      `e3v819c-20200807T1404-1509.mp4` the start time should be `14:09` not `14:04`. Use the 2nd value
      of the timestamp and count 1 hour back from that. Also note that this video file does not have
      the seconds listed. You will need to open this video and find the seconds in the timestamp in the
      video. In this example the video timestamp is `12:09:08 08/07/20`, just take the seconds, so the correct
      timestamp in this case is `14:09:08`. 
  - The difference in times in this example for CAF42 is `00:01:59` (1 min, 59 seconds), the video was started
    after the neural data, which is typical except in one or two exceptional cases.
- **Step 2:** Run `ntk_sync.map_video_to_neural_data` and set the parameter `manual_video_neural_offset_sec` to `119`. 
  Any time the video starts *after* the neural data (which is typical) the value will be positive, in the odd case
  when video started before neural data the parameter will be negative.

#### Example command line usage of ntk_sync.map_video_to_neural_data

Running save_map_video_to_neural_data from the command line produces an NPZ file (Numpy zip containing multiple data structures)

Example using CAF42:

```bash
python neuraltoolkit/ntk_sync.py save_map_video_to_neural_data \
    --output_filename "~/path/to/output/map_video_to_neural_data_CAF42.npz" \
    --syncpulse_files "/path/to/CAF42/SyncPulse/DigitalPanel*.bin" \
    --video_files "/path/to/CAF42/Video/e3*.mp4" \
    --neural_files "s3://hengenlab/CAF42/Neural_Data/Headstages*.bin" \
    --sleepstate_files "/path/to/CAF42/SleepState/*.npy" \
    --dlclabel_files "/path/to/CAF42/DeepLabCut/*.h5" \
    --neural_bin_files_per_sleepstate_file 288 \
    --manual_video_neural_offset_sec 119
```
Note that sleep only video, neural files, and digital files are required, other properties are optional. 
Run `python save_map_video_to_neural_data --help` for command line documentation. 

The NPZ file that is produces contains the following data:
- `output_matrix` - the main output of the function `ntk_sync.map_video_to_neural_data`, see the function documentation
  for a detailed explanation of the data structure.
- `video_files` - a list of the video filenames, the video files are listed by index number in `output_matrix`.
- `neural_files` - a list of the neural filenames, the neural files are listed by index number in `output_matrix`.
- `sleepstate_files` - a list of the sleep state files (if any were provided).
- `syncpulse_files` - a list of the digital channel files used to find the sync pulse.
- `dlclabel_files` - a list of the deep lab cut files used (if any were provided)
- `camera_pulse_output_matrix` - Not normally used, this is the output of `map_videoframes_to_syncpulse` and is more
  commonly used for debugging.
- `pulse_ix` - the index into `camera_pulse_output_matrix` where the Sync Pulse can be found 
  (this is the change in duty cycle that signifies the beginning of camera recording)

Example accessing the data above in Python:

```python
import numpy as np

z = np.load("~/path/to/output/map_video_to_neural_data_CAF42.npz")
print(z.files)
# ['output_matrix', 'video_files', 'neural_files', 'sleepstate_files', 'syncpulse_files', 'dlclabel_files', 'camera_pulse_output_matrix', 'pulse_ix']

output_matrix = z['output_matrix']
print(output_matrix[0:5])
# [(181591149000, 0, 0, 0, 0, 3475000, 1, [-1., -1., -1., -1., -1., -1.])
#  (181657789000, 1, 0, 1, 0, 3476666, 1, [-1., -1., -1., -1., -1., -1.])
#  (181724469000, 2, 0, 2, 0, 3478333, 1, [-1., -1., -1., -1., -1., -1.])
#  (181791149000, 3, 0, 3, 0, 3480000, 1, [-1., -1., -1., -1., -1., -1.])
#  (181857789000, 4, 0, 4, 0, 3481666, 1, [-1., -1., -1., -1., -1., -1.])]
```

#### Command line functionality
Note these functions are less commonly used, typically reserved for experienced users.
* `python ntk_sync.py --help` 
  Get command line help output.
* `python ntk_sync.py save_map_video_to_neural_data --syncpulse_files FILENAME_GLOBS --video_files FILENAME_GLOBS --neural_files FILENAME_GLOBS [--sleepstate_files FILENAME_GLOBS] [--dlclabel_files FILENAME_GLOBS] [--output_filename map_video_to_neural_data.npz] [--fs 25000] [--n_channels 256] [--manual_video_frame_offset 0] [--recording_config EAB40.cfg]`
  Calls save_map_video_to_neural_data(...) which saves the results of map_video_to_neural_data(...) to a NPZ file.
* `python ntk_sync.py save_neural_files_bom [--output_filename FILENAME.csv] [[--neural_files NEURAL_FILENAMES_GLOB] --neural_files ...]` 
  Advanced usage: produces a CSV containing the eCube timestamps of a set of neural files which can be used instead of passing the neural files to the functions below (useful when the neural files are large and possibly difficult to access on demand).

```
import neuraltoolkit as ntk

# map_video_to_neural_data example usage:
output_matrix, video_files, neural_files, sleepstate_files, syncpulse_files, dlclabel_files = \
    ntk.map_video_to_neural_data(
        syncpulse_files='EAB40Data/EAB40_Camera_Sync_Pulse/*.bin'
        video_files=['EAB40Data/EAB40_Video/3_29-4_02/*.mp4',
                     'EAB40Data/EAB40_Video/4_02-4_05/*.mp4'],
        neural_files='EAB40Data/EAB40_Neural_Data/3_29-4_02/*.bin',
        dlclabel_files='EAB40Data/EAB40_DLC_Labels/*.h5'
        sleepstate_files='EAB40Data/EAB40_Sleep_States/*.npy'
    )

# map_videoframes_to_syncpulse example usage:
output_matrix, pulse_ix, files = ntk.map_videoframes_to_syncpulse('EAB40_Dataset/CameraSyncPulse/*.bin')
```
---
## Legacy ecube functions

```
# filename
rawfile = 'neuraltoolkit/Headstages_64_Channels_int16_2018-04-06_10-01-57.bin'

# Get number of channels
print("Enter total number of channels : ")
number_of_channels = np.int16(eval(input()))

# Time and data
t, dgc = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, 'hs64')

# Time and data for multiple probes with same number of channels
hstype = ['Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KS_chmap']
nprobes = 4
# number_of_channels here is total number of channels in all probes (64 * nprobes = 256)
t, dgc = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, hstype, nprobes)

# plot raw data
ntk.plot_data(dgc, 0, 25000, 1)

# plot bandpassed data
ntk.plot_data(bdgc, 0, 25000, 1)

# plot data in channel list
l = np.array([5, 13, 31, 32, 42, 46, 47, 49, 51, 52, 53, 54 ])
ntk.plot_data_chlist(bdgc, 25000, 50000, l )

# Time and data from rawdata for nsec
# For single probe
tt, ddgc = ntk.load_raw_binary_gain_chmap_nsec(rawfile, number_of_channels, 'hs64', 25000, 2)

# For multiple probes
# hstype = ['Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KS_chmap']
# nprobes = 4
tt, ddgc = ntk.load_raw_binary_gain_chmap_nsec(rawfile, number_of_channels, hstype, 25000, 2, nprobes)

# Load preprocessed data file
pdata = ntk.load_raw_binary_preprocessed(preprocessedfilename, number_of_channels)

# Load one channel from data
number_of_channels = 64
channel_number = 4
# lraw is 1 for raw file and for prepocessed file lraw is 0
ch_data = ntk.load_a_ch(rawfile, number_of_channels, channel_number,
                    lraw=1)

# Load ecube data and returns time(if raw) and data in range
number_of_channels = 64
lraw is 1 for raw file and for prepocessed file lraw is 0
hstype,  linear if preprocessed file
ts = 0, start from begining of file or can be any sample number
te = 2500, read 2500 sample points from ts ( te greater than ts)
if ts =0 and te = -1,  read from begining to end of file
t, bdgc = ntk.load_raw_binary_gain_chmap_range(rawfile, number_of_channels,
                                           hstype, nprobes=1,
                                           lraw=1, ts=0, te=25000)
```
---


## FAQ
```
1. ImportError: bad magic number in 'neuraltoolkit': b'\x03\xf3\r\n'
Please go to neuraltoolkit folder and remove *.pyc files
```
---

## Issues

```Please slack Kiran ```
---
