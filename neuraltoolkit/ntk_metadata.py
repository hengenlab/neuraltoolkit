#MAKE METADATA.CSV

import numpy as np
import os
import csv
import tkinter
from tkinter import Tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

#ask user what directory we should be in (corresponding to that animal)
continue_to_directory = 'dummy'
while(continue_to_directory != '1'):
	continue_to_directory = input("Please select the directory of animal info sheet.  Press '1' to continue.\n")

# root = tkinter.Tk()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# dirname = filedialog.askdirectory(parent=root, initialdir="/",
#                                     title='Please select the raw data directory for the animal of interest.')
dirname = filedialog.askdirectory()
os.chdir(dirname)

#check if metadata.csv exists already.
if os.path.isfile('metadata.csv') == True:
	#load contents of metadata.csv as meta_dict
	with open('metadata.csv') as f:
	    meta_dict = dict(filter(None, csv.reader(f)))
	#then tell the user what is present in metadata.csv
	print('The following is already present in metadata.csv:\n')
	for i in meta_dict:
	    print(i + ": " + meta_dict[i])
elif os.path.isfile('metadata.csv') == False:
	meta_dict = dict()


input_section = 'dummyvar'

while (input_section != 'q'):
	input_section = input('''\n\nWould you like to enter information about\nanimal (1)\nsurgery (2)\nadditional surgery (3)\nrecording (4)\nvideo (5)\nexperimental details (6)\n
(Press 'q' to quit and save/discard changes). \n(Press 's' to scrape data from another animal sheet)\n(Press 'v' to view current metadata.)\n\nEnter: ''')

	if input_section == '1':
		# #ANIMAL INFO
		meta_dict['animal_id'] = input('Animal ID (ex EAB00022)\nEnter: ')
		meta_dict['strain'] = input('Strain (ex Long Evans)\nEnter: ')
		meta_dict['species'] = input('Species (rat or mouse)\nEnter: ')
		meta_dict['dob'] = input('Date of Birth (enter May 16 2018 as 051618)\nEnter: ')
		meta_dict['sex'] = input('Sex (M or F)\nEnter: ')

	elif input_section == '2':
		# #SURGERY INFO
		meta_dict['implantdate'] = input('Implant Date (enter May 16 2018 as 051618)\nEnter: ')
		meta_dict['hstype'] = input('''Headstage Type\nOptions are:\n
hs64\neibless-hs64_port32\neibless-hs64_port64\nintan32\nSi_64_KS_chmap\nSi_64_KT_T1_K2_chmap\n
Please enter exactly as above.\n
Enter: ''')

		# number of implant sites
		number_of_implants = input('How many implant sites does this animal have?\nEnter: ')

		# for each implant site, get target_site [eg v1], AP, ML, DV coordinates if user has them, and which channels correspond to that site

		for i in range(int(number_of_implants)):
			meta_dict['implantSite_'+str(i+1)] = input('Target Site #' + str(i+1) + ' [eg V1]\nEnter: ')
			meta_dict['APcoord_'+str(i+1)] = input('AP coordinates\nEnter: ')	
			meta_dict['MLcoord_'+str(i+1)] = input('ML coordinates\nEnter: ')
			meta_dict['DVcoord_'+str(i+1)] = input('DV coordinates\nEnter: ')
			meta_dict['site_'+str(i+1)+'_channels'] = input("Which channels are in this implant site? [eg '1-32']\nEnter: ")

		#EMG?
		emg_present = input('Is there an EMG? [If yes type 1.  If no type 0.] \nEnter: ')
		if emg_present == '1':
			meta_dict['emg_ch'] = input("What channel is the EMG on?\nIf more than one EMG, separate both channels by a space. [e.g. '1 2'] \nEnter: ")
		elif emg_present == '0':
			meta_dict['emg_ch'] = float('nan')

		# EEG?
		eeg_present = input('Is there an EEG? [If yes type 1.  If no type 0.] \nEnter: ')
		if eeg_present == '1':
			meta_dict['eeg_ch'] = input("What channel is the EEG on?\nIf more than one EEG, separate both channels by a space. [e.g. '1 2'] \nEnter: ")
		elif eeg_present == '0':
			meta_dict['eeg_ch'] = float('nan')

		#Surgery notes
		meta_dict['surgery_notes'] = input('Enter any other surgical notes\nEnter: ')

	elif input_section == '3':
	# # ADDITIONAL SURGERY INFO
		additional_surgery = input('Was there an additional surgery? [ex MD, viral injection, catheterization]\nIf yes type 1.  If no type 0.\nEnter: ')
		if additional_surgery == '1':
			meta_dict['add_surg_type'] = input('''What type of additional surgery was performed?\nOptions are:\n
MD\nviral_injection\ncatheter\nother\n
Please enter exactly as above.\n
Enter: ''')
			meta_dict['add_surg_date'] = input('When did the additional surgery take place? (enter May 16 2018 as 051618)\nEnter: ')
			if meta_dict['add_surg_type'] == 'MD':
				meta_dict['add_surg_site'] = input('On which side was MD? [ex L or R]\nEnter: ')
			if meta_dict['add_surg_type'] == 'viral_injection':
				meta_dict['add_surg_site'] = input('Where was the virus injected? [ex V1]\nEnter: ')
			if meta_dict['add_surg_type'] == 'catheter':
				meta_dict['add_surg_site'] = input('Where was catheter implanted? \nEnter: ')
			if meta_dict['add_surg_type'] == 'other':
				meta_dict['add_surg_notes']  = float('nan')
			meta_dict['add_surg_notes'] = input('Enter any other ' + meta_dict['add_surg_type'] + ' notes\nEnter: ')
		elif additional_surgery == '0':
			meta_dict['add_surg_date'] = float('nan')
			meta_dict['add_surg_type'] = float('nan')
			meta_dict['add_surg_site']  = float('nan')
			meta_dict['add_surg_notes']  = float('nan')

	elif input_section == '4':
	# #RECORDING INFO
	# need to make it so you can add more epochs

	# How many recording start/stops?
		number_of_restarts = input('How many times was the recording restarted?\nEnter: ')

		for i in range(int(number_of_restarts)+1):
			print('Entering information for recording epoch #' + str(i+1) + ':\n')
			# meta_dict['epoch_' + str(i+1) + '_lightdir'] = input("What is the directory that the digital input for lights was saved in for this epoch? (typically Digital Input 1)\nEnter: ")
			# meta_dict['epoch_' + str(i+1) + '_camdir'] = input("What is the directory that the digital input for cameras was saved in for this epoch? (typically Digital Input 2)\nEnter: ")

			light_key = 'dummy'
			while(light_key != '1'):
				light_key = input("What is the directory that the digital input for lights was saved in for this epoch? (typically Digital Input 1)  Press '1' to select file.\n")
			Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
			meta_dict['epoch_' + str(i+1) + '_lightdir'] = filedialog.askdirectory() # show an "Open" dialog box and return the path to the selected file

			cam_key = 'dummy'
			while(cam_key != '1'):
				cam_key = input("What is the directory that the digital input for cameras was saved in for this epoch? (typically Digital Input 2)  Press '1' to select file.\n")
			Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
			meta_dict['epoch_' + str(i+1) + '_camdir'] = filedialog.askdirectory() # show an "Open" dialog box and return the path to the selected file

			vid_key = 'dummy'
			while(vid_key != '1'):
				vid_key = input("What is the directory that video was saved in for this epoch?  Press '1' to select file.\n")
			Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
			meta_dict['epoch_' + str(i+1) + '_viddir'] = filedialog.askdirectory() # show an "Open" dialog box and return the path to the selected file							


			meta_dict['epoch_' + str(i+1) + '_port'] = input('Port # for recording epoch #' + str(i+1) + '\nEnter: ')
			meta_dict['epoch_' + str(i+1) + '_client'] = input('Client # for recording epoch #' + str(i+1) + '\nEnter: ')
			meta_dict['epoch_' + str(i+1) + '_start_date'] = input('Start date for recording epoch #' + str(i+1) + '. (enter May 16 2018 as 051618)\nEnter: ')
			meta_dict['epoch_' + str(i+1) + '_start_time'] = input("Start time for recording epoch #" + str(i+1) + ". (enter in military time, eg '15:04')\nEnter: ")
			meta_dict['epoch_' + str(i+1) + '_stop_date'] = input('Stop date for recording epoch #' + str(i+1) + '. (enter May 16 2018 as 051618)\nEnter: ')
			meta_dict['epoch_' + str(i+1) + '_stop_time'] = input("Stop time for recording epoch #" + str(i+1) + ". (enter in military time, eg '15:04')\nEnter: ")
			meta_dict['epoch_' + str(i+1) + '_bad_channels'] = input('''List all bad channels (seen in Open Ephys or NanoZ output).
If more than one bad channel, separate channels by a space. [e.g. '1 5 16']
Enter only 'broken' channels.  Not just silent channels.
Enter: ''')

		meta_dict['recording_notes'] = input('Enter any other recording notes\nEnter: ')


	elif input_section == '5':
		meta_dict['video_dir'] = input('What is the directory that the Watchtower videos are saved in?\nEnter: ')
		meta_dict['camera_IDs'] = input('Which camera(s) were used for this animal? e.g. 8107 812f\nEnter: ')

	elif input_section == '6':
		# #EXPERIMENTAL NOTES

		#TODO: add the following options:
		# Experimental treatment [eg. 'Injection']
		# if injection:
		# how many injs?
		# for each inj:
		# drug?
		# dose?
		# injtime?

		# Euthanization_date
		meta_dict['euthanization_date'] = input('When was the animal euthanized? (enter May 16 2018 as 051618)\nEnter: ')
		meta_dict['experimental_notes'] = input('Please enter any additional experimental notes\nEnter: ')

	elif input_section == 's':
		#scrape from Betsy's spreadsheet so she doesn't have to type it out again
		import pandas as pd
		import datetime

		################## Use below to prompt dialog box to select appropriate file ##################

		input_key = 'dummy'
		while(input_key != '1'):
			input_key = input("Please select the directory of animal info sheet.  Press '1' to continue.\n")

		Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

		# filename =  '/Users/sbrunwas/Box/fortytwo/Bradtke/Bradtke_Tracking.xlsx'
		df = pd.read_excel(io = filename)

		animalid = input('What is the animal ID?\n')


		if df['ANIMAL ID'].isin([animalid]).sum() == 1:
			print(animalid + " was succesfully found!")
		elif df['ANIMAL ID'].isin([animalid]).sum() == 0:
			print(animalid + " could not be found.")

		df[df['ANIMAL ID'] == animalid]['SEX'].iloc[0]


		# #ANIMAL INFO
		meta_dict['animal_id'] = df[df['ANIMAL ID'] == animalid]['ANIMAL ID'].iloc[0]
		meta_dict['strain'] = df[df['ANIMAL ID'] == animalid]['STRAIN'].iloc[0]
		meta_dict['species'] = df[df['ANIMAL ID'] == animalid]['SPECIES'].iloc[0]


		dob = df[df['ANIMAL ID'] == animalid]['DOB'].iloc[0]
		if (type(dob) == pd._libs.tslib.Timestamp) | (type(dob) == datetime.datetime):
			meta_dict['dob'] = str(dob.month) + str(dob.day) + str(dob.year)[2:4]
		else:
			meta_dict['dob'] = dob
		meta_dict['sex'] = df[df['ANIMAL ID'] == animalid]['SEX'].iloc[0]

		# #SURGERY INFO
		implantdate = df[df['ANIMAL ID'] == animalid]['IMPLANT DATE'].iloc[0]
		if (type(implantdate) == pd._libs.tslib.Timestamp) | (type(implantdate) == datetime.datetime):
			meta_dict['implantdate'] = str(implantdate.month) + str(implantdate.day) + str(implantdate.year)[2:4]
		else:
			meta_dict['implantdate'] = implantdate
		meta_dict['implantSite'] = df[df['ANIMAL ID'] == animalid]['PROJECT NUMBER'].iloc[0]
		meta_dict['APcoord'] = df[df['ANIMAL ID'] == animalid]['AP'].iloc[0]	
		meta_dict['MLcoord'] = df[df['ANIMAL ID'] == animalid]['ML'].iloc[0]
		meta_dict['DVcoord'] = df[df['ANIMAL ID'] == animalid]['DV'].iloc[0]

		#surgery missing info
		#hstype, number_of_implants, sites for each implant (if more than one), EMG, EEG, other surgical notes

		# # ADDITIONAL SURGERY INFO
		meta_dict['add_surg_type'] = df[df['ANIMAL ID'] == animalid]["ADDT'L SURGERY"].iloc[0]
		add_surg_date = df[df['ANIMAL ID'] == animalid]["ADDT'L SURGERY DATE"].iloc[0]
		if (type(add_surg_date) == pd._libs.tslib.Timestamp) | (type(add_surg_date) == datetime.datetime):
			meta_dict['add_surg_date'] = str(add_surg_date.month) + str(add_surg_date.day) + str(add_surg_date.year)[2:4]
		else:
			meta_dict['add_surg_date'] = add_surg_date

		# #RECORDING INFO
		recording_start = df[df['ANIMAL ID'] == animalid]['BEGIN RECORD'].iloc[0]
		if (type(recording_start) == pd._libs.tslib.Timestamp) | (type(recording_start) == datetime.datetime):
			meta_dict['recording_start'] = str(recording_start.month) + str(recording_start.day) + str(recording_start.year)[2:4]
		else:
			meta_dict['recording_start'] = recording_start

		recording_end = df[df['ANIMAL ID'] == animalid]['END RECORD'].iloc[0]
		if (type(recording_end) == pd._libs.tslib.Timestamp) | (type(recording_end) == datetime.datetime):
			meta_dict['recording_end'] = str(recording_end.month) + str(recording_end.day) + str(recording_end.year)[2:4]
		else:
			meta_dict['recording_end'] = recording_end

		meta_dict['recording_notes'] = df[df['ANIMAL ID'] == animalid]['NOTES'].iloc[0]

		#recording missing info
		# number_of_restarts, port, client, epoch start_date, epoch start_time, epoch stop_date, epoch stop_dimte, bad channels

		# Euthanization_date
		euthanization_date = df[df['ANIMAL ID'] == animalid]['EUTHANIZED'].iloc[0]
		if (type(euthanization_date) == pd._libs.tslib.Timestamp) | (type(euthanization_date) == datetime.datetime):
			meta_dict['euthanization_date'] = str(euthanization_date.month) + str(euthanization_date.day) + str(euthanization_date.year)[2:4]
		else:
			meta_dict['euthanization_date'] = euthanization_date


		print('\nThe following was found in ' + filename + ':\n')
		for i in meta_dict:
		    print(str(i) + ": " + str(meta_dict[i]))

	elif input_section == 'v':
		print('\nThe following metadata has been submitted.\n')
		for i in meta_dict:
		    print(str(i) + ": " + str(meta_dict[i]))
		print('\nNote: Metadata must still be saved upon exiting.\n')


	# #_of_clusters from kilosort (pulled from ks_out directory)


save_changes = input("Save changes to metadata.csv? ('Y' or 'N')\n")
if save_changes == 'Y':
	#to save meta_dict as .csv file in current directory
	with open('metadata.csv','w') as f:
	    w = csv.writer(f)
	    w.writerows(meta_dict.items())
	print('Changes to metadata.csv have been saved in current directory.')
elif save_changes == 'N':
	print('Changes to metadata.csv have been discarded.')
