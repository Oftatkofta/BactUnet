import unittest
import numpy as np
from window_functions import hanning_window, hamming_window, blackman_window, kaiser_window, bartlett_window, apply_window
import logging

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestWindowFunctions(unittest.TestCase):
    def test_hanning_window(self):
        # Test Hanning window generation
        N = 10  # Number of points in the window
        logging.debug(f"Testing Hanning window with N={N}")
        window = hanning_window(N)
        logging.debug(f"Generated Hanning window: {window}")
        self.assertEqual(len(window), N)  # Check if window length is correct
        self.assertTrue(np.all(window >= 0))  # Ensure all values are non-negative
        self.assertTrue(np.all(window <= 1))  # Ensure all values are within expected range

    def test_hamming_window(self):
        # Test Hamming window generation
        N = 10  # Number of points in the window
        logging.debug(f"Testing Hamming window with N={N}")
        window = hamming_window(N)
        logging.debug(f"Generated Hamming window: {window}")
        self.assertEqual(len(window), N)  # Check if window length is correct
        self.assertTrue(np.all(window >= 0))  # Ensure all values are non-negative
        self.assertTrue(np.all(window <= 1))  # Ensure all values are within expected range

    def test_blackman_window(self):
        # Test Blackman window generation
        N = 10  # Number of points in the window
        logging.debug(f"Testing Blackman window with N={N}")
        window = blackman_window(N)
        logging.debug(f"Generated Blackman window: {window}")
        self.assertEqual(len(window), N)  # Check if window length is correct
        self.assertTrue(np.all(window >= 0))  # Ensure all values are non-negative
        self.assertTrue(np.all(window <= 1))  # Ensure all values are within expected range

    def test_kaiser_window(self):
        # Test Kaiser window generation
        N = 10  # Number of points in the window
        beta = 5  # Shape parameter for the Kaiser window
        logging.debug(f"Testing Kaiser window with N={N} and beta={beta}")
        window = kaiser_window(N, beta)
        logging.debug(f"Generated Kaiser window: {window}")
        self.assertEqual(len(window), N)  # Check if window length is correct
        self.assertTrue(np.all(window >= 0))  # Ensure all values are non-negative
        self.assertTrue(np.all(window <= 1))  # Ensure all values are within expected range

    def test_bartlett_window(self):
        # Test Bartlett window generation
        N = 10  # Number of points in the window
        logging.debug(f"Testing Bartlett window with N={N}")
        window = bartlett_window(N)
        logging.debug(f"Generated Bartlett window: {window}")
        self.assertEqual(len(window), N)  # Check if window length is correct
        self.assertTrue(np.all(window >= 0))  # Ensure all values are non-negative
        self.assertTrue(np.all(window <= 1))  # Ensure all values are within expected range

    def test_apply_window_hanning(self):
        # Test applying a Hanning window to the data
        data = np.sin(2 * np.pi * np.linspace(0, 1, 10))  # Generate sine wave data
        logging.debug(f"Testing apply_window with Hanning window on data: {data}")
        windowed_data = apply_window(data, 'hanning')
        logging.debug(f"Windowed data (Hanning): {windowed_data}")
        self.assertEqual(len(windowed_data), len(data))  # Check if windowed data length matches original data
        self.assertTrue(np.all(windowed_data <= 1))  # Ensure all values are within expected range
        self.assertTrue(np.all(windowed_data >= -1))  # Ensure all values are within expected range

    def test_apply_window_kaiser(self):
        # Test applying a Kaiser window to the data
        data = np.sin(2 * np.pi * np.linspace(0, 1, 10))  # Generate sine wave data
        beta = 5  # Shape parameter for the Kaiser window
        logging.debug(f"Testing apply_window with Kaiser window (beta={beta}) on data: {data}")
        windowed_data = apply_window(data, 'kaiser', beta=beta)  # Apply Kaiser window with beta=5
        logging.debug(f"Windowed data (Kaiser, beta={beta}): {windowed_data}")
        self.assertEqual(len(windowed_data), len(data))  # Check if windowed data length matches original data
        self.assertTrue(np.all(windowed_data <= 1))  # Ensure all values are within expected range
        self.assertTrue(np.all(windowed_data >= -1))  # Ensure all values are within expected range

    def test_apply_window_invalid_type(self):
        # Test applying a window with an invalid type
        data = np.ones(10)  # Generate constant data
        logging.debug(f"Testing apply_window with invalid window type on data: {data}")
        with self.assertRaises(ValueError) as context:
            apply_window(data, 'invalid')  # Attempt to apply an invalid window type
        logging.debug(f"Caught exception: {context.exception}")
        self.assertEqual(str(context.exception), "Unknown window type: invalid")  # Check if correct error message is raised

    def test_apply_window_invalid_data(self):
        # Test applying a window with invalid data types
        invalid_data_types = [
            [1, 2, 3, 4, 5],  # List instead of NumPy array
            "invalid_data",   # String
            {1: "a", 2: "b"}, # Dictionary
            12345              # Integer
        ]
        for data in invalid_data_types:
            logging.debug(f"Testing apply_window with invalid data type: {data}")
            with self.assertRaises(TypeError):
                apply_window(data, 'hanning')  # Attempt to apply a Hanning window to invalid data

if __name__ == '__main__':
    unittest.main()