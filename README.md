# Neuraltoolkit

## Installation

### Download neuraltoolkit
git clone https://github.com/hengenlab/neuraltoolkit.git 
Enter your username and password

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




## load ecube data
#### List of functions
 
* load_raw_binary                 : To load plain raw data
* load_raw_binary_gain            : To load raw data with gain
* load_raw_binary_gain_chmap      : To load raw data with gain and channel mapping
* load_raw_binary_gain_chmap_nsec : To load nsec of raw data with gain and channel mapping

```
import neuraltoolkit as ntk
import numpy as np
from matplotlib import pyplot as plt

# Get filename
rawfile = 'neuraltoolkit/Headstages_64_Channels_int16_2018-04-06_10-01-57.bin'

# Get number of channels
print("Enter total number of channels : ")
number_of_channels = np.int16(eval(input()))

# Time and data
t, dgc = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, 'hs64')

# Time only
t = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, 'hs64', t_only=1)

# Time and data for multiple probes with same number of channels
hstype = ['Si_64_KS_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KT_T1_K2_chmap', 'Si_64_KS_chmap']
nprobes = 4
# number_of_channels here is total number of channels in all probes (64 * nprobes = 256)
t, dgc = ntk.load_raw_binary_gain_chmap(rawfile, number_of_channels, hstype, nprobes)

# bandpass filter
bdgc = ntk.butter_bandpass(dgc, 500, 7500, 25000, 3)

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

# Load digital data for cameras, etc
tdig, ddig = ntk.load_digital_binary(digitalrawfile)

# Load time only from digital data for cameras, etc
tdig = ntk.load_digital_binary(digitalrawfile, t_only=1)

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
```

## load intan data
#### List of functions
* load_intan_raw_gain_chanmap	: To load raw data with gain and channel mapping

```
# import libraries
import neuraltoolkit as ntk
import numpy as np
from matplotlib import pyplot as plt

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
```

## video
#### List of functions/Class
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

```

## filters
#### List of functions
* butter_bandpass
* butter_highpass
* butter_lowpass
* welch_power
* notch_filter
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
```

## high dimensional data
#### List of functions

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
#### List of functions

```
# import libraries
import neuraltoolkit as ntk
import numpy as np
from matplotlib import pyplot as plt

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
