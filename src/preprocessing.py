import numpy as np
import cv2
import logging
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to normalize image data to the range [0, 1]
def normalize_image(image):
    """
    Normalize the pixel values of an image to the range [0, 1].

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The normalized image.
    """
    epsilon = 1e-8  # Small value to prevent division by zero
    logging.debug(f"Normalizing image with shape {image.shape}")
    # Check if image has non-zero range to avoid zero division
    if np.ptp(image) == 0:
        logging.warning("Image has no range. Returning a zeroed image.")
        return np.zeros_like(image)
    # Normalize the image by subtracting the minimum and dividing by the range
    return (image - np.min(image)) / (np.ptp(image) + epsilon)

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian blur to an image to reduce noise.

    Parameters:
    image (numpy.ndarray): The input image.
    kernel_size (tuple): The size of the Gaussian kernel.
    sigma (float): The standard deviation for the Gaussian kernel. If sigma is set to 0, it will be calculated automatically based on the kernel size.

    Returns:
    numpy.ndarray: The blurred image.
    """
    logging.debug(f"Applying Gaussian blur with kernel size {kernel_size} and sigma {sigma}")
    # Ensure kernel size is valid (must be odd, positive, and non-negative)
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0 or kernel_size[0] <= 0 or kernel_size[1] <= 0:
        logging.error("Kernel size must be odd and positive. Please provide a valid kernel size.")
        raise ValueError("Kernel size must be odd and positive.")
    # Apply Gaussian blur using the specified kernel size and sigma value
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Function to resize an image to a specified size
def resize_image(image, size=(128, 128)):
    """
    Resize an image to the given size.

    Parameters:
    image (numpy.ndarray): The input image.
    size (tuple): The desired size (width, height).

    Returns:
    numpy.ndarray: The resized image.
    """
    logging.debug(f"Resizing image from shape {image.shape} to size {size}")
    # Check if the desired size is valid
    if any(dim <= 0 for dim in size):
        logging.error("Size values must be positive. Please provide a valid size.")
        raise ValueError("Size values must be positive.")
    # Resize the image using linear interpolation for better quality
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

# Function to apply thresholding to an image
def apply_threshold(image, threshold_value=127):
    """
    Apply a binary threshold to an image.

    Parameters:
    image (numpy.ndarray): The input image.
    threshold_value (int): The threshold value.

    Returns:
    numpy.ndarray: The thresholded image.
    """
    logging.debug(f"Applying threshold with value {threshold_value}")
    # Ensure the image is in a valid format for thresholding
    if image.dtype == np.float32 or image.dtype == np.float64:
        logging.warning("Converting image from float format to 8-bit format for thresholding.")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        logging.error("Unsupported image format for thresholding. Expected uint8 or float.")
        raise TypeError("Unsupported image format for thresholding. Expected uint8 or float.")
    # Apply binary thresholding to the image
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

# Function to perform edge detection using Canny
def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection to an image.

    Parameters:
    image (numpy.ndarray): The input image.
    threshold1 (int): The first threshold for the hysteresis procedure.
    threshold2 (int): The second threshold for the hysteresis procedure.

    Returns:
    numpy.ndarray: The image with edges detected.
    """
    logging.debug(f"Applying Canny edge detection with thresholds {threshold1} and {threshold2}")
    # Ensure the image is in 8-bit format for edge detection
    if image.dtype == np.float32 or image.dtype == np.float64:
        logging.warning("Converting image from float format to 8-bit format for Canny edge detection.")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        logging.error("Unsupported image format for Canny edge detection. Expected uint8 or float.")
        raise TypeError("Unsupported image format for Canny edge detection. Expected uint8 or float.")
    # Use Canny edge detection to find edges in the image
    return cv2.Canny(image, threshold1, threshold2)

# Example usage
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image preprocessing script")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    args = parser.parse_args()

    # Read an image from file
    logging.info(f"Reading image from file '{args.image_path}'")
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully loaded
    if image is None:
        logging.error("The image file could not be loaded. Please check the file path and try again.")
        print("Error: The image file could not be loaded. Please check the file path and try again.")
        return

    # Normalize the image to the range [0, 1]
    logging.info("Normalizing the image")
    normalized_image = normalize_image(image)
    
    # Apply Gaussian blur to reduce noise
    logging.info("Applying Gaussian blur to the image")
    blurred_image = apply_gaussian_blur(normalized_image)

    # Resize the image to a fixed size of 128x128 pixels
    logging.info("Resizing the image")
    resized_image = resize_image(blurred_image)

    # Apply binary thresholding to the resized image
    logging.info("Applying thresholding to the image")
    thresholded_image = apply_threshold(resized_image)

    # Apply Canny edge detection to find edges in the thresholded image
    logging.info("Applying Canny edge detection to the image")
    edges = apply_canny_edge_detection(thresholded_image)

    # Display the results using OpenCV windows
    logging.info("Displaying the results")
    cv2.imshow('Original Image', image)  # Original grayscale image
    cv2.imshow('Normalized Image', normalized_image)  # Normalized image
    cv2.imshow('Blurred Image', blurred_image)  # Blurred image
    cv2.imshow('Resized Image', resized_image)  # Resized image
    cv2.imshow('Thresholded Image', thresholded_image)  # Thresholded image
    cv2.imshow('Edges', edges)  # Edges detected using Canny
    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()