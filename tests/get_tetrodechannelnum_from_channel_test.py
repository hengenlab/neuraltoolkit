import unittest
import neuraltoolkit as ntk


class TestGetTetrodechannelnumFromChannel(unittest.TestCase):

    def test_default_group_size(self):
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(0), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(1), 1)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(2), 2)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(3), 3)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(4), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(8), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(60), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(63), 3)

    def test_custom_group_size(self):
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(10, 3), 1)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(15, 5), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(9, 6), 3)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(14, 7), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(15, 5), 0)
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(16, 5), 1)

    def test_channel_zero(self):
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(0), 0)

    def test_group_size_one(self):
        self.assertEqual(ntk.get_tetrodechannelnum_from_channel(10, 1), 0)

    def test_negative_channel(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrodechannelnum_from_channel(-1)

    def test_zero_group_size(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrodechannelnum_from_channel(10, 0)

    def test_negative_group_size(self):
        with self.assertRaises(ValueError):
            ntk.get_tetrodechannelnum_from_channel(10, -1)


if __name__ == '__main__':
    unittest.main()
