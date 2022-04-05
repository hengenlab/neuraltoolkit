import unittest
import numpy as np
import neuraltoolkit as ntk


class Testfind_channel_map(unittest.TestCase):
    expected_output = \
        np.asarray([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,
                    4,  28, 32, 24, 20,
                    48,  44, 36, 40, 64, 60, 52, 56, 54, 50, 42,
                    46, 62, 58, 34, 38,
                    39,  35, 59, 63, 47, 43, 51, 55, 53, 49, 57,
                    61, 37, 33, 41, 45,
                    17,  21, 29, 25, 1,  5, 13, 9,  11, 15, 23,
                    19, 3,  7,  31, 27]) - 1

    number_of_channels = None

    def test_find_channel_map(self):
        test_output = \
                ntk.find_channel_map('hs64', self.number_of_channels)
        msg = "check channel map"
        self.assertEqual(self.expected_output.tolist(),
                         test_output.tolist(), msg)


class Testchannel_map_data(unittest.TestCase):
    hsch = \
        np.asarray([26, 30, 6,  2,  18, 22, 14, 10, 12, 16, 8,
                    4,  28, 32, 24, 20,
                    48,  44, 36, 40, 64, 60, 52, 56, 54, 50, 42,
                    46, 62, 58, 34, 38,
                    39,  35, 59, 63, 47, 43, 51, 55, 53, 49, 57,
                    61, 37, 33, 41, 45,
                    17,  21, 29, 25, 1,  5, 13, 9,  11, 15, 23,
                    19, 3,  7,  31, 27]) - 1
    expected_output = np.concatenate((hsch, hsch+hsch.size), axis=0)
    data = np.arange(0, 128, 1)
    data = np.tile(data, (128, 4))
    data = data.T
    number_of_channels = 128
    hstype = ['hs64', 'hs64']
    nprobes = 2

    def test_channel_map_data(self):
        test_output =\
            ntk.channel_map_data(self.data, self.number_of_channels,
                                 self.hstype, nprobes=self.nprobes)
        msg = "failed channel_map_data"
        self.assertEqual(self.expected_output.tolist(),
                         test_output[:, 0].tolist(),
                         msg)
        self.assertEqual(self.expected_output.tolist(),
                         test_output[:, 1].tolist(),
                         msg)
        self.assertEqual(self.expected_output.tolist(),
                         test_output[:, 2].tolist(),
                         msg)
        self.assertEqual(self.expected_output.tolist(),
                         test_output[:, 3].tolist(),
                         msg)
