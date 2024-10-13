import numpy as np
import matplotlib.pyplot as plt
from window_functions import hanning_window, hamming_window, blackman_window, kaiser_window, bartlett_window, apply_window

def kaiser_window_fixed(N):
    # Generate a Kaiser window with a fixed beta value of 5
    return kaiser_window(N, beta=5)

def kaiser_window_2d(M, N, beta=5):
    """
    Generate a 2D Kaiser window.
    
    Parameters:
    M, N : int
        Number of rows and columns in the output window.
    beta : float, optional
        Shape parameter for the Kaiser window (default is 5).
    
    Returns:
    window : ndarray
        The 2D Kaiser window.
    """
    # Create 1D Kaiser windows for each dimension
    window_m = np.kaiser(M, beta)
    window_n = np.kaiser(N, beta)
    
    # Create 2D window using outer product
    window_2d = np.outer(window_m, window_n)
    
    return window_2d

def demo_2d_kaiser_window():
    """
    Demonstrate the 2D Kaiser window with different beta values.
    """
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Generate 2D Kaiser windows with different beta values
    beta_values = [1, 5, 10, 20]
    for i, beta in enumerate(beta_values):
        window = kaiser_window_2d(100, 100, beta=beta)
        
        # Plot the 2D window
        im = axs[i].imshow(window, cmap='viridis')
        axs[i].set_title(f'2D Kaiser Window (beta={beta})')
        fig.colorbar(im, ax=axs[i])

    plt.tight_layout()
    plt.show()

def main():
    # Generate sample data: a sine wave with 100 data points
    t = np.linspace(0, 1, 100)  # Generate 100 points between 0 and 1 (time vector)
    data = np.sin(2 * np.pi * 5 * t)  # Create a sine wave with a frequency of 5 Hz

    # Create a figure with subplots to plot original and windowed data
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # Create a 3x2 grid of subplots
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    # Plot original data
    axs[0].plot(t, data, label='Original Data')  # Plot the original sine wave
    axs[0].set_title('Original Sine Wave')  # Set the title for the subplot
    axs[0].set_xlabel('Time')  # Label the x-axis
    axs[0].set_ylabel('Amplitude')  # Label the y-axis
    axs[0].legend()  # Add a legend to the plot

    # Apply different windows and plot the results
    windows = {
        'hanning': hanning_window,
        'hamming': hamming_window,
        'blackman': blackman_window,
        'kaiser': kaiser_window_fixed,
        'bartlett': bartlett_window
    }

    # Iterate over each window type and apply it to the sine wave
    for i, (window_type, window_func) in enumerate(windows.items(), start=1):
        window = window_func(len(data))  # Generate the window using the appropriate function
        windowed_data = apply_window(data, window_type, beta=5 if window_type == 'kaiser' else None)  # Apply the window to the data
        
        # Plot the windowed data in the corresponding subplot
        axs[i].plot(t, windowed_data, label=f'{window_type.capitalize()} Windowed Data')
        axs[i].set_title(f'{window_type.capitalize()} Window Applied to Sine Wave')  # Set the title for the subplot
        axs[i].set_xlabel('Time')  # Label the x-axis
        axs[i].set_ylabel('Amplitude')  # Label the y-axis
        axs[i].legend()  # Add a legend to the plot

    # Add a call to the 2D Kaiser window demo
    demo_2d_kaiser_window()

    # Adjust layout to prevent overlapping elements and show all plots simultaneously
    plt.tight_layout()
    plt.show()

    # Add a call to the 2D Kaiser window demo
    demo_2d_kaiser_window()

if __name__ == "__main__":
    main()
