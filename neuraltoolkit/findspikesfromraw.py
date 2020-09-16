import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt

# rawfile = '/Volumes/bs001r/users/EAB_09-12/EAB_00010_2018-06-08_15-06-33/Headstages_64_Channels_int16_2018-06-10_11-21-42.bin'
# number_of_channels = 64
# hs = 'hs64'
# nsec = 1
# number_of_probes = 1
# ch_list = [43,44,63]
# thresh = -50

# Get filename
print("Enter filename ")
rawfile = str(input())
print(rawfile)

# Get number of ptonrd
print("Enter total number of probes: ")
number_of_probes = np.int16(eval(input()))
print(number_of_probes)

# Get number of channels
print("Enter total number of channels : ")
number_of_channels = np.int16(eval(input()))
print(number_of_channels)

if number_of_probes > 1:
    hs = []
    for i in range(number_of_probes):
        # Get number of channels
        print("Enter probe type (Ex. hs64) : ")
        hstype = input()
        print(hstype)
        hs.append(hstype)
else:
    # Get number of channels
    print("Enter probe type (Ex. hs64) : ")
    hs = input()
    print(hs)

print("hstype ", hs)

# Get number of seconds
print("Enter total number of seconds to plot : ")
nsec = np.int16(eval(input()))
print(nsec)

#Get channel numbers
print('Enter channels to plot: ')
print('Ex: 32 35 48')
ch_list = np.asarray(input().split(' '),dtype='int')  
print(ch_list)

# Get threshold
print("What threshold to use for spikes?  (Recommended: -70)")
thresh = np.int16(eval(input()))
print(thresh)

# get data
tt, ddgc = ntk.load_raw_binary_gain_chmap_nsec(rawfile, number_of_channels, hs,
                                               25000, nsec, number_of_probes)
# bandpass data
bdgc = ntk.butter_bandpass(ddgc, 500, 7500, 25000, 3)

plt.figure(1)
for i in range(len(ch_list)):
    # print(i, " ", ch_list[i])
    ax = plt.subplot(len(ch_list), 1, i+1)
    plt.plot(bdgc[ch_list[i], :])
    plt.xticks([])
    # plt.yticks([])
    plt.title('Ch '+ str(ch_list[i]+1))

bdgc_thres = bdgc
bdgc_thres[bdgc_thres>thresh] = 0

plt.figure(2)
for i in range(len(ch_list)):
    # print(i, " ", ch_list[i])
    ax = plt.subplot(len(ch_list), 1, i+1)
    plt.plot(bdgc_thres[ch_list[i], :])
    plt.xticks([])
    # plt.yticks([])
    plt.title('Ch '+ str(ch_list[i]+1))

bdgc_grad = bdgc_thres-np.roll(bdgc_thres,1)
bdgc_grad[bdgc_grad==0] = 1000
bdgc_grad[bdgc_grad!=1000] = -1000
bdgc_grad[bdgc_grad==1000] = 1
bdgc_grad[bdgc_grad==(-1000)] = 0

plt.figure(3)
for i in range(len(ch_list)):
    # print(i, " ", ch_list[i])
    ax = plt.subplot(len(ch_list), 1, i+1)
    plt.plot(bdgc_grad[ch_list[i], :])
    plt.xticks([])
    # plt.yticks([])
    plt.title('Ch '+ str(ch_list[i]+1))
plt.show()

#need to fix this
# spiketimes = np.where(bdgc_grad==1)