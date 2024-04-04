import unittest
import os
import os.path as op
import numpy as np
import neuraltoolkit as ntk

# # Load only one probe from raw file
# rawfile = '/home/kbn/Headstages_512_Channels_int16_2019-06-21_03-55-09.bin'
# number_of_channels = 512
# hstype = ['EAB50chmap_00', 'EAB50chmap_00', 'EAB50chmap_00',
#           'EAB50chmap_00', 'EAB50chmap_00', 'EAB50chmap_00',
#           'EAB50chmap_00', 'EAB50chmap_00']   # Channel map
# ts = 0, start from begining of file or can be any sample number
# te = 2500, read 2500 sample points from ts ( te greater than ts)
# if ts =0 and te = -1,  read from begining to end of file
# nprobes = 8
# probenum = 0  # which probe to return (starts from zero)
# probechans = 64  #  number of channels per probe (symmetric)
# t,dgc = ntk.load_raw_gain_chmap_1probe(rawfile, number_of_channels,
#                                        hstype, nprobes=nprobes,
#                                        lraw=1, ts=0, te=-1,
#                                        probenum=0, probechans=64)
#


class Testload_raw_gain_chmap_1probe(unittest.TestCase):
    # os.chdir('/hlabhome/kiranbn/git/neuraltoolkit/tests')
    # bdir = os.path.join(os.path.dirname(__file__), 'data')
    print(os.getcwd())
    if op.exists('/home/runner/work/neuraltoolkit/neuraltoolkit/tests/'):
        os.chdir('/home/runner/work/neuraltoolkit/neuraltoolkit/tests/')
    hs = np.asarray([2, 5, 1, 22, 9, 14, 18, 47, 23, 26, 31, 3, 35, 4,
                     7, 16, 34, 21, 12, 10, 29, 17, 8, 13, 11, 6, 38,
                     19, 24, 20, 15, 25, 37, 32, 28, 27, 52, 46, 41,
                     30, 61, 57, 54, 33, 55, 43, 63, 36, 58, 51, 60,
                     42, 40, 50, 64, 48, 59, 49, 44, 45, 62, 56, 53,
                     39]) - 1
    gain = np.float64(0.19073486328125)
    expected_output_t = [np.uint64(np.loadtxt('timestamp.csv',
                                              delimiter=','))]
    expected_output = np.loadtxt('data.csv', delimiter=',')
    expected_output = np.asarray(np.int16(expected_output[0:64, :] * gain))
    expected_output = expected_output[hs, :]
    # expected_output_l = np.asarray(np.loadtxt('tests/data/data.csv',
    expected_output_l = np.asarray(np.loadtxt('data.csv',
                                              delimiter=','))
    print("sh expected_output_l ", expected_output_l.shape)
    expected_output_l = np.asarray(np.int16(expected_output_l[-64:, :] *
                                            gain))
    # expected_output_l = expected_output[hs, :]
    rawfile = 'Headstages_512_Channels_int16_2021-06-08_11-08-03.bin'
    number_of_channels = 512
    hstype = ['APT_PCB', 'APT_PCB', 'APT_PCB', 'APT_PCB',
              'APT_PCB', 'APT_PCB', 'APT_PCB', 'APT_PCB']
    hstype_l = ['linear', 'linear', 'linear', 'linear',
                'linear', 'linear', 'linear', 'linear']
    ts = 0
    te = -1
    nprobes = 8
    probenum = 0
    probechans = 64
    probenum_l = 7
    expected_noise_output = \
        np.loadtxt('remove_large_noise.csv', delimiter=',')

    def test_channel_map_data(self):
        test_output_t = None
        test_output = None
        test_output_t, test_output =\
            ntk.load_raw_gain_chmap_1probe(self.rawfile,
                                           self.number_of_channels,
                                           self.hstype,
                                           nprobes=self.nprobes,
                                           lraw=1,
                                           ts=self.ts,
                                           te=self.te,
                                           probenum=self.probenum,
                                           probechans=self.probechans)
        print("expected_output_t ", self.expected_output_t)
        print("test_output_t ", test_output_t)
        print("expected_output ", self.expected_output)
        print("test_output ", test_output)
        msg = "check time in load_raw_gain_chmap_1probe"
        self.assertEqual(self.expected_output_t,
                         test_output_t.tolist(), msg)
        msg = "check data in load_raw_gain_chmap_1probe"
        self.assertEqual(self.expected_output.tolist(),
                         test_output.tolist(),
                         msg)

        # test remove_large_noise
        checkchans = np.array([0, 1])
        max_value_to_check = 135
        _, test_noise_output = (
            ntk.remove_large_noise(
                test_output,
                max_value_to_check=max_value_to_check,
                windowval=25000,
                checkchans=checkchans,
                lplot=0
            )
        )
        msg = "check ntk remove_large_noise"
        self.assertEqual(self.expected_noise_output.tolist(),
                         test_noise_output.tolist(),
                         msg)

    def test_channel_map_data_l(self):
        test_output_t = None
        test_output = None
        test_output_t, test_output =\
            ntk.load_raw_gain_chmap_1probe(self.rawfile,
                                           self.number_of_channels,
                                           self.hstype_l,
                                           nprobes=self.nprobes,
                                           lraw=1,
                                           ts=self.ts,
                                           te=self.te,
                                           probenum=self.probenum_l,
                                           probechans=self.probechans)
        print("expected_output_t ", self.expected_output_t)
        print("test_output_t ", test_output_t)
        print("expected_output_l ", self.expected_output_l)
        print("test_output ", test_output)
        msg = "check time in load_raw_gain_chmap_1probe"
        self.assertEqual(self.expected_output_t,
                         test_output_t.tolist(), msg)
        msg = "check data in load_raw_gain_chmap_1probe"
        self.assertEqual(self.expected_output_l.tolist(),
                         test_output.tolist(),
                         msg)
