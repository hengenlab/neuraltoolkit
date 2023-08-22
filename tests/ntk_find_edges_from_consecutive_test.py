import unittest
import neuraltoolkit as ntk


class Test_ntk_find_edges_from_consecutive(unittest.TestCase):
    inputs = [[0, 47, 48, 49, 50, 97, 98, 99],
              [2020, 2019, 2018, 2017, 2015, 2013, 2012, 2011, 2010]]

    expected_output = \
        [[[0, 0], [47, 50], [97, 99]],
         [[2010, 2013], [2015, 2015], [2017, 2020]]]

    def test_find_edges_from_consecutive(self):
        for indx in range(len(self.inputs)):
            test_output = \
                    ntk.find_edges_from_consecutive(self.inputs[indx],
                                               step=1, lverbose=0)
            msg = \
                f'edges {self.expected_output[indx]} {test_output}'
            self.assertEqual(self.expected_output[indx],
                             test_output, msg)
