import neuraltoolkit as ntk
import glob
import os.path as op
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

DATA_DIR = '/media/disk1/ABC12345/ABC12345_L4_W3_2025-10-23_10-05-53/'
# Get Headstage files and check samples from time and filesize
print('\n\n\n\nCheck Headstage files now')
fl_list = ntk.natural_sort(glob.glob(op.join(DATA_DIR, 'H*.bin')))
number_of_channels = 64
hstype = ['APT_PCB']
lb1 = 1
lb2 = 1
for binfile1, binfile2 in zip(fl_list[0:-1], fl_list[1:]):
    samples = ntk.samples_between_two_binfiles(binfile1, binfile2,
                                               number_of_channels,
                                               hstype, nprobes=1,
                                               lb1=lb1, lb2=lb1,
                                               fs=25000)
    samples2 = ntk.find_samples_per_chan(binfile1, number_of_channels,
                                         lraw=lb1)
    # print(f'Headstagefiles: samples {samples} {samples2} {samples2-samples}')
    diff = int(samples2 - samples)
    status = "OK" if abs(diff) <= 1 else "DIFFERENT"
    print(f"Headstage files: samples {samples} {samples2} {diff} → {status}")

# Get Analog files and check samples from time and filesize
print('\n\n\n\nCheck Analog files now')
fl_list = ntk.natural_sort(glob.glob(op.join(DATA_DIR, 'A*.bin')))
number_of_channels = 10
hstype = ['APT_PCB']
lb1 = 0
lb2 = 0
for binfile1, binfile2 in zip(fl_list[0:-1], fl_list[1:]):
    samples = ntk.samples_between_two_binfiles(binfile1, binfile2,
                                               number_of_channels,
                                               hstype, nprobes=1,
                                               lb1=lb1, lb2=lb1,
                                               fs=25000)
    samples2 = ntk.find_samples_per_chan(binfile1, number_of_channels,
                                         lraw=lb1)
    # print(f'Analog files: samples {samples} {samples2} {samples2-samples}')
    diff = int(samples2 - samples)
    status = "OK" if abs(diff) <= 1 else "DIFFERENT"
    print(f"Analog files: samples {samples} {samples2} {diff} → {status}")

# Get Digital files and check samples from time and filesize
print('\n\n\n\nCheck Analog files now')
fl_list = ntk.natural_sort(glob.glob(op.join(DATA_DIR, 'D*.bin')))
#
# size of int64 / size of int16 =  8 bytes/2 bytes = 4
number_of_channels = 4

hstype = ['APT_PCB']
lb1 = 0
lb2 = 0
for binfile1, binfile2 in zip(fl_list[0:-1], fl_list[1:]):
    samples = ntk.samples_between_two_binfiles(binfile1, binfile2,
                                               number_of_channels,
                                               hstype, nprobes=1,
                                               lb1=lb1, lb2=lb1,
                                               fs=25000)
    samples2 = ntk.find_samples_per_chan(binfile1, number_of_channels,
                                         lraw=lb1)
    # print(f'Digital files: samples {samples} {samples2} {samples2-samples}')
    diff = int(samples2 - samples)
    status = "OK" if abs(diff) <= 1 else "DIFFERENT"
    print(f"Digital files: samples {samples} {samples2} {diff} → {status}")
