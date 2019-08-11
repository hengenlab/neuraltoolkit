""" Test cases can be run with pytest ntk_ecube_test.py """
import unittest
import tempfile
import numpy as np
import neuraltoolkit as ntk


class MapVideoframesToSyncpulseTest(unittest.TestCase):
    input_tests = [
        # Test case #1
        [
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        ],
        # Test case #2
        [
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],  # file 1
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # file 2
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],        # file 3
        ],
    ]

    expected_results = [
        # Expected result #1
        [
            [-2, 0,  0, 2],
            [-1, 0,  2, 2],
            [-2, 0,  4, 2],
            [-1, 0,  6, 2],
            [-2, 0,  8, 3],
            [0,  0, 11, 5],
            [-2, 0, 16, 1],
            [1,  0, 17, 2],
            [-2, 0, 19, 2],
        ],
        # Expected result #2
        [
            [-2, 0,  0, 2],
            [-1, 0,  2, 2],
            [-2, 0,  4, 2],
            [-1, 0,  6, 2],
            [-2, 0,  8, 3],
            [0,  0, 11, 5],
            [-2, 0, 16, 1],
            [1,  0, 17, 2],
            [-2, 0, 19, 4],
            [2,  1,  2, 2],
            [-2, 1,  4, 2],
            [3,  1,  6, 2],
            [-2, 1,  8, 3],
            [4,  1, 11, 5],
            [-2, 1, 16, 1],
            [5,  1, 17, 4],
            [-2, 2,  0, 2],
            [6,  2,  2, 2],
            [-2, 2,  4, 2],
            [7,  2,  6, 2],
            [-2, 2,  8, 3],
            [8,  2, 11, 5],
            [-2, 2, 16, 1],
            [9,  2, 17, 2],
        ],
    ]

    def test_map_videoframes_to_syncpulse(self):
        for i, input_test in enumerate(MapVideoframesToSyncpulseTest.input_tests):
            with tempfile.TemporaryDirectory() as td:
                for j, each_file in enumerate(input_test):
                    f = open(td + '/sync_pulse_{}.bin'.format(j), mode='w')
                    np.array(0, dtype=np.int64).tofile(f)
                    np.array(each_file, dtype='int64').tofile(f)
                    f.close()

                output_matrix, _, _ = ntk.map_videoframes_to_syncpulse(td + '/*')
                self.assertTrue(np.all(output_matrix == np.array(MapVideoframesToSyncpulseTest.expected_results[i])))
