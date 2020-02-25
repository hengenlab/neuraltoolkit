# This code will serve as the backbone for everyone projects
# Capability of this code is to pull spiketimes from certain days and waveform template(s)
# It will also be able to plot crude binned firing rate 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy import io
import matplotlib
import glob
import os
import random
import math
import scipy.stats as stats

class neuron(object):
	'''Create instances (objects) of the class neuron ''' 
	def __init__(self, animaldir = False, cell_number = False, fs = 25000, start_day = False, end_day = False): 
		print('working on neuron '+str(cell_number))
		# going to folder that contains processed data
		os.chdir(animaldir)
		# loading relevant numpy files
		spikefiles = np.sort(glob.glob("*spike_times.npy"))
		clustfiles = np.sort(glob.glob("*spike_clusters.npy"))
		#peakfiles = np.sort(glob.glob("*peakchannel.npy"))
		wfs = np.load('temwf.npy')

		if end_day-start_day > 1:
			keys = np.load(glob.glob("new_key*")[0])
			length_start = [(spikefiles[i].find('length_')+7) for i in range(start_day, end_day)]
			length_end = [(spikefiles[i].find('_p')) for i in range(start_day, end_day)]
			length = [int(spikefiles[i][length_start[i]:length_end[i]]) for i in range(start_day, end_day)]
			length = np.append([0], length)
			length = np.cumsum(length)
		else:
			length = [0]

		print('loading clusters')
		curr_clust = [np.load(clustfiles[i]) for i in range(start_day, end_day)]

		print('loading spikes')
		curr_spikes = [np.load(spikefiles[i])+length[i] for i in range(start_day, end_day)]
		#peak_chans  = np.concatenate([np.load(peakfiles[i])for i in range(start_day, end_day)])

		#self.peak_chans = np.zeros(start_day-end_day)
		
		if end_day-start_day > 1:
			spiketimes = []	
			for f in range(start_day, end_day):
				clusters = np.load('unique_clusters.npy')
				key_val = keys[f, int(cell_number-1)]
				clust_idx = clusters[int(key_val)]
				spk_idx = np.where(curr_clust[f] == clust_idx)[0]
				spks = np.concatenate(curr_spikes[f][spk_idx])/fs 
				#self.peak_chans[f] = peak_chans[f][int(key_val)]
				spiketimes.append(spks)
			spiketimes = np.concatenate(spiketimes)
			
		else:
			clusters = np.load('unique_clusters.npy')
			clust_idx = clusters[int(cell_number-1)]
			spk_idx = np.where(curr_clust[0] == clust_idx)[0]
			spiketimes = curr_spikes[0][spk_idx]/fs
			#self.peak_chans = peak_chans[int(cell_number-1)]

		self.time = np.concatenate(spiketimes)
		self.wf_temp = wfs[cell_number-1]


	def plot_wf(self):

		with sns.axes_style("white"):
			fig1 = plt.figure()
			plt.plot(self.wf_temp, color = '#8fb67b')
			plt.xlabel('Time')
			plt.ylabel('Voltage (uV)')
		sns.despine()


	def firing_rate(self, binsz = 1800, start = 0, end = False):

		# This will produce a firing rate plot for all loaded spike times unless otherwise specified
		# binsz, start, end are in seconds 

		if end == False:
			end = max(self.time)
		edges   = np.arange(start,end, binsz)
		bins    = np.histogram(self.time,edges)
		hzcount = bins[0]
		hzcount = hzcount/binsz
		#hzcount[hzcount==0] = 'NaN'
		xbins   = bins[1]
		xbins   = xbins/3600

		fig1        = plt.figure()
		plt.ion()

		with sns.axes_style("white"):
		    plt.plot(xbins[:-1],hzcount, color = '#703be7')

		plt.ion()
		sns.set()
		sns.despine()
		plt.gca().set_xlabel('Time (hours)')
		plt.gca().set_ylabel('Firing rate (Hz)')
		plt.show()

	def isi_hist(self, start = 0, end = False, isi_thresh = 0.1, nbins = 101):
		# This plot produces a histogram of ISIs and tells you ISI contamination
		# You can adjust start and end time (in seconds)
		# You can adjust threshold for ISIs in the plot (in seconds) and the number of bins used
		if end == False:
			end = max(self.time)
		idx = np.where(np.logical_and(self.time>=start, self.time<=end))[0]
		ISI = np.diff(self.time[idx])
		edges = np.linspace(0,isi_thresh,nbins)
		hist_isi        = np.histogram(ISI,edges)
		contamination   = 100*(sum(hist_isi[0][0:int((0.1/isi_thresh)*(nbins-1)/50)])/sum(hist_isi[0]))
		contamination   = round(contamination,2)
		cont_text       = 'Contamination is {} percent.' .format(contamination)

		plt.ion()
		with sns.axes_style("white"):
		    fig1            = plt.figure()  
		    ax              = fig1.add_subplot(111)
		    ax.bar(edges[1:]*1000-0.5, hist_isi[0],color='#6a79f7')
		    ax.set_ylim(bottom = 0)
		    ax.set_xlim(left = 0)
		    ax.set_xlabel('ISI (ms)')
		    ax.set_ylabel('Number of intervals')
		    ax.text(30,0.7*ax.get_ylim()[1],cont_text)
		sns.despine()
		return ISI









