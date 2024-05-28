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


class TestGetTetrodeChannelsFromChannelnum(unittest.TestCase):

    def test_default_group_size(self):
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(2),
                         [0, 1, 2, 3])
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(5),
                         [4, 5, 6, 7])
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(63),
                         [60, 61, 62, 63])

    def test_custom_group_size(self):
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(2,
                         ch_grp_size=5), [0, 1, 2, 3, 4])
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(7,
                         ch_grp_size=5), [5, 6, 7, 8, 9])
        self.assertEqual(ntk.get_tetrode_channels_from_channelnum(79,
                         ch_grp_size=5), [75, 76, 77, 78, 79])

    def test_invalid_channel_number(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrode_channels_from_channelnum(-1)
