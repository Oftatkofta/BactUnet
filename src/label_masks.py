import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tifffile import TiffFile, imwrite
from scipy.ndimage import label, minimum_filter1d

def min_projection(image_stack):
    """
    Apply a minimum filter projection across a moving window of three frames.

    Parameters:
    image_stack (numpy.ndarray): A 3D NumPy array representing the stack of images.

    Returns:
    numpy.ndarray: A 3D array where each frame is the minimum projection of the original frames.
    """
    # Using scipy.ndimage.minimum_filter1d to apply a minimum filter across the frames
    # This effectively reduces noise by taking the minimum value over a window of three frames
    output = minimum_filter1d(image_stack, size=3, axis=0)

    return output

def label_array(image_stack):
    """
    Label the connected features in each frame of the image stack.

    Parameters:
    image_stack (numpy.ndarray): A 3D NumPy array representing the stack of images.

    Returns:
    numpy.ndarray: A labeled version of the input stack where connected features are uniquely numbered.
    """
    # Initialize the output array with zeros to avoid using uninitialized memory
    labeled_stack = np.zeros_like(image_stack)

    # Iterate through each frame in the image stack
    for i in range(len(image_stack)):
        # Label connected features in the current frame
        labeled_frame, num_features = label(image_stack[i])
        # Store the labeled frame in the output stack
        labeled_stack[i] = labeled_frame

    return labeled_stack

def process_tiff_file(input_path, output_path):
    """
    Process a TIFF file by applying minimum projection and labeling connected features.

    Parameters:
    input_path (str): Path to the input TIFF file.
    output_path (str): Path to save the processed TIFF file.
    """
    try:
        # Read the TIFF file
        with TiffFile(input_path) as tif:
            # Check if the TIFF file has any pages (frames)
            if not tif.pages:
                raise ValueError(f"The TIFF file '{input_path}' is empty or could not be read correctly.")
            # Convert the TIFF file to a NumPy array
            image_stack = tif.asarray()

        # Apply minimum projection to reduce noise and label connected features
        min_proj = min_projection(image_stack)
        labeled_stack = label_array(min_proj)

        # Save the processed stack to a new TIFF file
        imwrite(output_path, labeled_stack)
        print(f"Processed file saved to '{output_path}'")

    # Handle specific exceptions for better error reporting
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except OSError as oe:
        print(f"OS error: {oe}")
    except Exception as e:
        print(f"An unexpected error occurred while processing file '{input_path}': {e}")

def main():
    """
    Main function to execute the TIFF processing workflow.

    This function handles command-line arguments for specifying the input and output paths.
    """
    # Ensure that the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python label_masks.py <input_tiff_file> <output_tiff_file>")
        sys.exit(1)

    # Get the input and output file paths from the command-line arguments
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Check if the input file exists before proceeding
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    # Process the input TIFF file and save the output
    process_tiff_file(input_path, output_path)

# Entry point of the script
if __name__ == "__main__":
    main()