import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob

tik = time.time()

filedir = '/media/bs006r/EAB00050/EAB00050_2019-10-11_16-30-45_p10_c3/'
# filedir='/media/bs006r/EAB00050/EAB00050_2019-10-11_16-30-45_p10_c3/block1/'
os.chdir(filedir)
binfiles = np.sort(glob.glob('*.bin'))

# list_of_files = binfiles[0:3]
list_of_files = binfiles[0:20]
time_of_files = []
spiketimes = [np.array([0])]*64

number_of_probes = 8
number_of_channels = 512
# hs = ['linear']*number_of_probes
# nsec = 300
thresh = -70

for j,file in enumerate(list_of_files):
	print(str('Working on file ' + str(j+1) + ' of ' + str(len(list_of_files)) + ' files'))
	print(file)
	tt, ddgc = ntk.load_raw_binary_gain(file,number_of_channels)
	ddgc = ddgc[0:64,:]
	bdgc = ntk.butter_bandpass(ddgc, 500, 7500, 25000, 3)
	bdgc_thres = bdgc.copy()
	bdgc_thres[bdgc_thres>thresh] = 0
	bdgc_grad = bdgc_thres-np.roll(bdgc_thres,1)
	bdgc_grad[bdgc_grad>0] = 1000
	bdgc_grad[bdgc_grad!=1000] = -1000
	bdgc_grad[bdgc_grad==1000] = 1
	bdgc_grad[bdgc_grad==(-1000)] = 0

	offset = np.sum(time_of_files)
	
	time_of_files.append(bdgc_grad.shape[1])

	spiketimes_file = []
	for i in np.arange(64):
	    spiketimes_file.append(np.where(bdgc_grad[i,:]==1))

	spiketimes = [np.append(spiketimes[i],spiketimes_file[i]+offset) for i in range(len(spiketimes))]

tok = time.time()
print(str('Time to process spiketimes: ' + str(tok-tik)))


###PLOT SPIKETIMES AS ARRAY

binsize = 60 #seconds
binrange = np.arange(0,np.sum(np.array(time_of_files)),step = (binsize*25000)+1)

spiketimes_array = np.zeros([64,binrange.shape[0]-1])
for i in np.arange(64):
    counts,bins = np.histogram(spiketimes[i],bins=binrange)
    spiketimes_array[i,:] = counts

plt.imshow(spiketimes_array)
plt.xlabel('Time (minutes)')
plt.ylabel('Channel')

figurename = str('EAB50' + '_' +  list_of_files[0][-23:-4] + '__' + list_of_files[-1][-23:-4] + '.png')
arrayname = str('EAB50' + '_' +  list_of_files[0][-23:-4] + '__' + list_of_files[-1][-23:-4] + '.npy')

plt.savefig(figurename)

np.save(arrayname,spiketimes_array)




