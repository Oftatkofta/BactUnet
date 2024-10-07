import numpy as np
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def hanning_window(N):
    """
    Generate a Hanning window.

    Parameters:
    N (int): Number of points in the window.

    Returns:
    numpy.ndarray: Hanning window of length N.
    """
    logging.debug("Generating Hanning window with N=%d", N)
    return np.hanning(N)

def hamming_window(N):
    """
    Generate a Hamming window.

    Parameters:
    N (int): Number of points in the window.

    Returns:
    numpy.ndarray: Hamming window of length N.
    """
    logging.debug("Generating Hamming window with N=%d", N)
    return np.hamming(N)

def blackman_window(N):
    """
    Generate a Blackman window.

    Parameters:
    N (int): Number of points in the window.

    Returns:
    numpy.ndarray: Blackman window of length N.
    """
    logging.debug("Generating Blackman window with N=%d", N)
    return np.blackman(N)

def kaiser_window(N, beta):
    """
    Generate a Kaiser window.

    Parameters:
    N (int): Number of points in the window.
    beta (float): Shape parameter of the Kaiser window.

    Returns:
    numpy.ndarray: Kaiser window of length N.
    """
    logging.debug("Generating Kaiser window with N=%d and beta=%f", N, beta)
    return np.kaiser(N, beta)

def bartlett_window(N):
    """
    Generate a Bartlett window.

    Parameters:
    N (int): Number of points in the window.

    Returns:
    numpy.ndarray: Bartlett window of length N.
    """
    logging.debug("Generating Bartlett window with N=%d", N)
    return np.bartlett(N)

def apply_window(data, window_type, **kwargs):
    """
    Apply a window function to a given dataset.

    Parameters:
    data (numpy.ndarray): The data to which the window function is to be applied.
    window_type (str): The type of window to apply ('hanning', 'hamming', 'blackman', 'kaiser', 'bartlett').
    kwargs: Additional parameters for specific window types (e.g., 'beta' for 'kaiser').

    Returns:
    numpy.ndarray: The windowed data.
    """
    logging.debug("Applying window: %s", window_type)
    # Ensure the data is a NumPy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy.ndarray")

    N = len(data)
    logging.debug("Data length: %d", N)

    # Dictionary mapping window types to their respective functions
    window_functions = {
        'hanning': hanning_window,
        'hamming': hamming_window,
        'blackman': blackman_window,
        'kaiser': lambda N: kaiser_window(N, kwargs.get('beta', 14)),
        'bartlett': bartlett_window
    }

    # Get the appropriate window function from the dictionary
    if window_type not in window_functions:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Generate the window using the selected function
    window = window_functions[window_type](N)
    logging.debug("Window generated: %s", window)

    # Apply the window to the data element-wise
    windowed_data = data * window
    logging.debug("Windowed data: %s", windowed_data)
    return windowed_data

def main():
    """
    Main function to demonstrate the use of window functions.
    """
    # Generate sample data as a sine wave to better illustrate the effect of windowing
    t = np.linspace(0, 1, 100)  # Generate 100 points between 0 and 1
    data = np.sin(2 * np.pi * 5 * t)  # Generate a sine wave with frequency 5 Hz
    logging.info("Generated sine wave data: %s", data)

    # Apply Hanning window to the data
    hanning_data = apply_window(data, 'hanning')
    logging.info("Hanning window applied: %s", hanning_data)

    # Apply Kaiser window with beta = 10 to the data
    kaiser_data = apply_window(data, 'kaiser', beta=10)
    logging.info("Kaiser window applied with beta=10: %s", kaiser_data)

# Entry point of the script
if __name__ == "__main__":
    main()