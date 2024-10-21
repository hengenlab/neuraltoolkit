import unittest
import numpy as np
from neuraltoolkit import data_intpl


class TestDataIntpl(unittest.TestCase):

    def test_linear_interpolation(self):
        """ Test simple linear interpolation """
        tvec = np.array([0, 1, 2, 3])
        dvec = np.array([0, 1, 0, -1])
        nfact = 2

        # Run the function
        tvec_intpl, dvec_intpl = \
            data_intpl(tvec, dvec, nfact, intpl_kind='linear')

        # Test if the interpolated tvec is correct
        expected_tvec_intpl = np.linspace(tvec[0], tvec[-1], len(tvec) * nfact)
        np.testing.assert_almost_equal(tvec_intpl, expected_tvec_intpl,
                                       decimal=5)

        # Interpolate using scipy to get expected values for data
        interp_func = np.interp(tvec_intpl, tvec, dvec)
        np.testing.assert_almost_equal(dvec_intpl, interp_func, decimal=5)

    def test_cubic_interpolation(self):
        """ Test cubic interpolation """
        tvec = np.array([0, 1, 2, 3])
        dvec = np.array([0, 1, 0, -1])
        nfact = 3

        # Run the function with cubic interpolation
        tvec_intpl, dvec_intpl = \
            data_intpl(tvec, dvec, nfact, intpl_kind='cubic')

        # Check if interpolated values are computed (shape check)
        self.assertEqual(len(tvec_intpl), len(tvec) * nfact)
        self.assertEqual(len(dvec_intpl), len(tvec) * nfact)

    def test_invalid_nfact(self):
        """ Test invalid interpolation factor (nfact) """
        tvec = np.array([0, 1, 2])
        dvec = np.array([0, 1, 0])
        nfact = -1  # Invalid factor

        with self.assertRaises(ValueError):
            data_intpl(tvec, dvec, nfact)

    def test_invalid_input_length(self):
        """ Test mismatch between tvec and dvec lengths """
        tvec = np.array([0, 1, 2])
        dvec = np.array([0, 1])  # Length mismatch

        with self.assertRaises(ValueError):
            data_intpl(tvec, dvec, 2)

    def test_invalid_interpolation_kind(self):
        """ Test invalid interpolation kind """
        tvec = np.array([0, 1, 2])
        dvec = np.array([0, 1, 0])
        nfact = 2

        with self.assertRaises(ValueError):
            data_intpl(tvec, dvec, nfact, intpl_kind='invalid')

    def test_constant_data(self):
        """ Test interpolation with constant data """
        tvec = np.array([0, 1, 2, 3])
        dvec = np.array([5, 5, 5, 5])  # Constant data
        nfact = 2

        # Interpolated data should remain constant
        tvec_intpl, dvec_intpl = data_intpl(tvec, dvec, nfact)

        # Use assert_allclose to allow for small floating-point differences
        np.testing.assert_allclose(dvec_intpl, np.full_like(dvec_intpl, 5),
                                   rtol=1e-7, atol=1e-8)

    def test_single_point(self):
        """ Test interpolation with a single point (edge case) """
        tvec = np.array([0])
        dvec = np.array([1])
        nfact = 2

        # Interpolated values should be the same since there's only one point
        tvec_intpl, dvec_intpl = data_intpl(tvec, dvec, nfact)

        np.testing.assert_array_equal(tvec_intpl, np.array([0]))
        np.testing.assert_array_equal(dvec_intpl, np.array([1]))
