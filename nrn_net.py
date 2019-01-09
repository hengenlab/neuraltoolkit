from numpy import inf
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.patches   as patches
import math
import numpy.matlib         as matlib
import seaborn              as sns
import pandas               as pd
from scipy.stats import sem
from scipy import stats 
import warnings
import os
import pdb
import copy
import datetime as dt
import time


class neuron_net(object):

	def __init__(self, neurons):
		print(np.shape(neurons))
		self.neurons = [neurons[i] for i in range(0, len(neurons))]

	def plot_FR(self, binsz = 1800, start = 0, end = False):
		if end == False:
			end = max(self.time)
		fig1    = plt.figure()
		for i in self.neurons:

			# Plot the firing rate of the neuron in 1h bins, and add grey bars to indicate dark times. This is all based on the standard approach for our recordings and will have to be updated to accomodate any majorly different datasets (e.g. very short recordings or other L/D arrangements)
			edges   = np.arange(0,np.max(i.time)+12*3600,3600)
			bins    = np.histogram(i.time,edges)
			hzcount = bins[0]
			hzcount = hzcount/3600
			hzcount[hzcount==0] = 'NaN'
			#hzcount[0:48] = 'nan'
			xbins   = bins[1]
			xbins   = xbins/3600

			plt.ion()
			with sns.axes_style("white"):
			    plt.plot(xbins[:-1],hzcount)

			plt.ion()
			sns.set()
			sns.despine()
			plt.gca().set_xlabel('Time (hours)')
			plt.gca().set_ylabel('Firing rate (Hz)')
			plt.show()
