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
    # Define patch sizes
    patch_sizes = [(8, 8), (288, 288)]

    for patch_size in patch_sizes:
        # Generate sample data: a 2D sine wave
        x = np.linspace(0, 1, patch_size[0])
        y = np.linspace(0, 1, patch_size[1])
        X, Y = np.meshgrid(x, y)
        data = np.sin(2 * np.pi * 5 * X) * np.sin(2 * np.pi * 5 * Y)

        # Create a figure with subplots to plot original and windowed data
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.flatten()

        # Plot original data
        im = axs[0].imshow(data, cmap='viridis')
        axs[0].set_title(f'Original {patch_size[0]}x{patch_size[1]} Patch')
        fig.colorbar(im, ax=axs[0])

        # Apply different windows and plot the results
        windows = {
            'hanning': hanning_window,
            'hamming': hamming_window,
            'blackman': blackman_window,
            'kaiser': kaiser_window,
            'bartlett': bartlett_window,
            'kaiser_fixed': kaiser_window_fixed,
            'rectangular': lambda x: np.ones(x),
            'triangular': lambda x: np.bartlett(x)
        }

        # Iterate over each window type and apply it to the data
        for i, (window_type, window_func) in enumerate(windows.items(), start=1):
            if window_type == 'kaiser':
                window = window_func(patch_size[0], patch_size[1], beta=5)
            elif window_type in ['rectangular', 'triangular']:
                window = np.outer(window_func(patch_size[0]), window_func(patch_size[1]))
            else:
                window = np.outer(window_func(patch_size[0]), window_func(patch_size[1]))
            
            windowed_data = data * window
            
            im = axs[i].imshow(windowed_data, cmap='viridis')
            axs[i].set_title(f'{window_type.capitalize()} Window')
            fig.colorbar(im, ax=axs[i])

        plt.tight_layout()
        plt.show()

    # Add a call to the 2D Kaiser window demo
    demo_2d_kaiser_window()

if __name__ == "__main__":
    main()