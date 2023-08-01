import unittest
import neuraltoolkit as ntk


class Test_tetrode_channels(unittest.TestCase):
    expected_output = [[0, 1, 2, 3],
                       [0, 1, 2, 3],
                       [0, 1, 2, 3],
                       [0, 1, 2, 3],
                       [56, 57, 58, 59],
                       [60, 61, 62, 63],
                       [60, 61, 62, 63]]

    def test_tetrode_channels_from_channelnum(self):
        for indx, ch in enumerate([0, 1, 2, 3, 59, 60, 63]):
            test_output = \
                    ntk.get_tetrode_channels_from_channelnum(ch, 4)
            msg = \
                f'get_tetrode_channels_from_channelnum {ch} {test_output}'
            self.assertEqual(self.expected_output[indx],
                             test_output, msg)
