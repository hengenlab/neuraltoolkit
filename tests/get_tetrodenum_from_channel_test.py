import unittest
import neuraltoolkit as ntk


class TestGetTetrodenumFromChannel(unittest.TestCase):

    def test_standard_cases(self):
        self.assertEqual(ntk.get_tetrodenum_from_channel(0), 0)
        self.assertEqual(ntk.get_tetrodenum_from_channel(3), 0)
        self.assertEqual(ntk.get_tetrodenum_from_channel(4), 1)
        self.assertEqual(ntk.get_tetrodenum_from_channel(7), 1)
        self.assertEqual(ntk.get_tetrodenum_from_channel(8), 2)
        self.assertEqual(ntk.get_tetrodenum_from_channel(63), 15)

    def test_custom_channel_group_size(self):
        self.assertEqual(ntk.get_tetrodenum_from_channel(0, ch_grp_size=5), 0)
        self.assertEqual(ntk.get_tetrodenum_from_channel(4, ch_grp_size=5), 0)
        self.assertEqual(ntk.get_tetrodenum_from_channel(5, ch_grp_size=5), 1)
        self.assertEqual(ntk.get_tetrodenum_from_channel(9, ch_grp_size=5), 1)
        self.assertEqual(ntk.get_tetrodenum_from_channel(10, ch_grp_size=5), 2)
        self.assertEqual(ntk.get_tetrodenum_from_channel(59, ch_grp_size=5),
                         11)

    def test_negative_channel_number(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrodenum_from_channel(-1)

    def test_large_channel_number(self):
        self.assertEqual(ntk.get_tetrodenum_from_channel(1000, ch_grp_size=10),
                         100)


if __name__ == '__main__':
    unittest.main()
