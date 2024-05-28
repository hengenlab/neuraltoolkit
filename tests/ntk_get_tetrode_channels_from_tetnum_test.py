import unittest
import neuraltoolkit as ntk


class TestGetTetrodeChannels(unittest.TestCase):

    def test_get_tetrode_channels_default_group_size(self):
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(0), [0, 1, 2, 3])
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(1), [4, 5, 6, 7])
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(2),
                         [8, 9, 10, 11])
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(15),
                         [60, 61, 62, 63])

    def test_get_tetrode_channels_custom_group_size(self):
        self.assertEqual(
            ntk.get_tetrode_channels_from_tetnum(0, ch_grp_size=5),
            [0, 1, 2, 3, 4]
        )
        self.assertEqual(
            ntk.get_tetrode_channels_from_tetnum(1, ch_grp_size=5),
            [5, 6, 7, 8, 9]
        )
        self.assertEqual(
            ntk.get_tetrode_channels_from_tetnum(15, ch_grp_size=5),
            [75, 76, 77, 78, 79]
        )

    def test_get_tetrode_channels_invalid_tetrode_num(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrode_channels_from_tetnum(-1)
        with self.assertRaises(ValueError):
            ntk.get_tetrode_channels_from_tetnum(16)

    def test_get_tetrode_channels_edge_cases(self):
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(0),
                         [0, 1, 2, 3])
        self.assertEqual(ntk.get_tetrode_channels_from_tetnum(15),
                         [60, 61, 62, 63])


if __name__ == "__main__":
    unittest.main()
