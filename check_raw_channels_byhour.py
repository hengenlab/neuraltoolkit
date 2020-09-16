import neuraltoolkit as ntk
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import gc
import sys


#Path to binary files to generate heat map for:
filedir = '/media/bs005r/CAF00026/CAF00026_2020-07-16_09-12-16/'
#Path to save files to 
savedir = '/media/bs005r/CAF00026/CAF00026_2020-07-16_09-12-16/'

os.chdir(filedir)
binfiles = np.sort(glob.glob('*.bin'))

# filepathtxt = '/media/HlabShare/Clustering_Data/CAF00026/717-720/file_path_name_block1.txt'
# filelist = open(filepathtxt,'r')
# content = filelist.read()
# binfiles = content.split('\n')

os.chdir(savedir)

#The number of 5 minute files you want to process (Recommend testing on a couple first to make sure its running
# and data isn't corrupted before processing larger chunks!

hourchunks = np.arange(0,144,12)

for hour, chunk in enumerate(hourchunks):
	tik = time.time()

	print("Working on hour " + str(hour+1))

	list_of_files = binfiles[chunk:(chunk+12)]
	#list_of_files = binfiles[0:100]
	time_of_files = []
	spiketimes = [np.array([0])]*64
	#print('spiketimes',spiketimes)

	number_of_probes = 3
	number_of_channels = 192
	# hs = ['linear']*number_of_probes
	# nsec = 300

	# Verify that threshold is correct by plotting a couple files first (you just want to make sure there isn't too much
	# noise in the heat plot).
	thresh = -50

	# Select the probe you want to plot (probenumber = 0, will plot first 64 channels). This is hard coded right now, will
	# need to be changed in the future.
	probenumber = 2 # starts from zero

	for j,file_name in enumerate(list_of_files):
		print(str('Working on file ' + str(j+1) + ' of ' + str(len(list_of_files)) + ' files'))
		print(file_name)
		tik1 = time.time()
		try:
			tt, ddgc = ntk.load_raw_binary_gain(file_name, number_of_channels)
		except:
			print("failed loading raw binary gain")
		tok1 = time.time()
		print(str('Time to load data: ' + str(tok1-tik1)))
		#ddgc = ddgc[0:64,:]
		tik2 = time.time()
		#print('probenumber ', probenumber)
		#print('ddgc.shape ', ddgc.shape)
		#print('pb n ', probenumber*64, (probenumber+1)*64)
		bdgc = ntk.butter_bandpass(ddgc[probenumber*64:(probenumber+1)*64,:], 500, 7500, 25000, 2) #changed from 3rd order bandpasss to 2nd order bandpass (500, 7500, 25000, 2) to (500,7500,25000,3)
		tok2 = time.time()

		print(str('Time to apply bandpass filter: ' + str(tok2-tik2)))
		# bdgc_thres = bdgc.copy()
		bdgc[bdgc>thresh] = 0
		bdgc_grad = bdgc-np.roll(bdgc,1)
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

		del ddgc
		del bdgc 
		gc.collect()	

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

	figurename = str('CAF26' + '_' +  list_of_files[0][-23:-4] + '__' + list_of_files[-1][-23:-4] + str('_') + str(probenumber+1) + '.png')
	arrayname = str('CAF26' + '_' +  list_of_files[0][-23:-4] + '__' + list_of_files[-1][-23:-4] +  str('_') + str(probenumber+1) + '.npy')

	plt.savefig(figurename)

	# np.save(arrayname,spiketimes_array)

