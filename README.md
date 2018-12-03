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

# bandpass filter
bdgc = ntk.butter_bandpass(dgc, 500, 7500, 25000, 3)

# plot raw data
ntk.plot_data(dgc, 0, 25000, 1)

# plot bandpassed data
ntk.plot_data(bdgc, 0, 25000, 1)
```

## load intan data


## filters
#### List of functions
* butter_bandpass
* butter_highpass
* butter_lowpass

```
# import libraries
import neuraltoolkit as ntk
import numpy as np
from matplotlib import pyplot as plt

# load raw data
rawdata = np.load('P_Headstages_64_Channels_int16_2018-11-15_14-30-49_t_2936265174075_l_7500584_p_0_u_0_chg_1.npy')

# bandpass filter
help(ntk.butter_bandpass)
result = ntk.butter_bandpass(rawdata, 500, 4000, 25000, 3)

# Plot result
plt.plot(result[1,0:25000])
plt.show()
```
