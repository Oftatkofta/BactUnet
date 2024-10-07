import numpy as np
import matplotlib.pyplot as plt
from window_functions import hanning_window, hamming_window, blackman_window, kaiser_window, bartlett_window, apply_window

def kaiser_window_fixed(N):
    # Generate a Kaiser window with a fixed beta value of 5
    return kaiser_window(N, beta=5)

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

    # Adjust layout to prevent overlapping elements and show all plots simultaneously
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()