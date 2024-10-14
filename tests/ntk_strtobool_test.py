import unittest
import neuraltoolkit as ntk


class TestStrToBool(unittest.TestCase):

    def test_strtobool_true_values(self):
        """Test all expected True values."""
        true_values = ['y', 'yes', 't', 'true', 'on', '1']
        for val in true_values:
            self.assertTrue(ntk.strtobool(val))

    def test_strtobool_false_values(self):
        """Test all expected False values."""
        false_values = ['n', 'no', 'f', 'false', 'off', '0']
        for val in false_values:
            self.assertFalse(ntk.strtobool(val))

    def test_strtobool_case_insensitivity(self):
        """Test that the function is case-insensitive."""
        self.assertTrue(ntk.strtobool('YES'))
        self.assertFalse(ntk.strtobool('No'))
        self.assertTrue(ntk.strtobool('TRUE'))
        self.assertFalse(ntk.strtobool('false'))

    def test_strtobool_invalid_values(self):
        """Test invalid values raise a ValueError."""
        invalid_values = ['maybe', '2', 'random', '', 'null']
        for val in invalid_values:
            with self.assertRaises(ValueError) as context:
                ntk.strtobool(val)
            self.assertIn("Invalid truth value", str(context.exception))
