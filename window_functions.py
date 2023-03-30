import numpy as np
import math


def hann_window(height, width):
    """
    https://doi.org/10.1371/journal.pone.0229839
    :param i: height in pixels
    :param j: width in pixels
    :return: nupy array of shape (i, j) of a Hann window
    """
    out = np.empty((height, width), np.float32)
    I = 2*math.pi/(height-1)
    J = 2*math.pi/(width-1)
    for i in range(height):
        for j in range(width):
            out[i,j] = 0.25 * (1-math.cos(I*i)) * (1-math.cos(J*j))

    return out

def corner_hann_window(height, width):
    """
    :param i: height in pixels
    :param j: width in pixels
    :return: nupy array of shape (i, j) of a upper left corner Hann window
    """
    out = np.empty((height, width), np.float32)
    I = 2*math.pi/(height-1)
    J = 2*math.pi/(width-1)
    for i in range(height):
        for j in range(width):
            if (i <= height/2) and (j <= width/2):
                out[i, j] = 1
            elif (i > height/2) and (j < width/2):
                out[i, j] = 0.5 * (1 - math.cos(I * i))
            elif (i < height/2) and (j > width/2):
                out[i,j] = 0.5 * (1 - math.cos(J*j))
            else:
                out[i,j] = 0.25 * (1-math.cos(I*i)) * (1-math.cos(J*j))

    return out

def top_hann_window(height, width):
    """
    :param i: height in pixels
    :param j: width in pixels
    :return: nupy array of shape (i, j) of a upper edge Hann window
    """
    out = np.empty((height, width), np.float32)
    I = 2*math.pi/(height-1)
    J = 2*math.pi/(width-1)
    for i in range(height):
        for j in range(width):
            if (i < height/2):
                out[i, j] = 0.5 * (1 - math.cos(J * j))
            else:
                out[i,j] = 0.25 * (1-math.cos(I*i)) * (1-math.cos(J*j))

    return out

def bartley_hann_window(height, width):
    """
    :param i: height in pixels
    :param j: width in pixels
    :return: nupy array of shape (i, j) of a Bartley-Hann window
    """
    out = np.empty((height, width), np.float32)

    a0 = 0.62
    a1 = 0.48
    a2 = 0.38

    I = 2*math.pi/(height)
    J = 2*math.pi/(width)

    for i in range(height):
        for j in range(width):
            out[i,j] = (a0 + a1 * abs((i / height) - 0.5) - a2 * math.cos(I * i)) * (a0 + a1 * abs((j / width) - 0.5) - a2 * math.cos(J * j))

    return out

def triangular_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a triangular window
    """
    out = np.empty((height, width), np.float32)
    I = 2/height
    J = 2/width

    for i in range(height):
        for j in range(width):
            out[i,j] = (1-abs(I*i -1)) * (1 - abs(J*j - 1))

    return out

def step_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a step window
    """
    
    return height

#def _maskgen_helper(i, j, n_side, window_type):
    #returns correct window generator


def build_weighted_mask_array(window_type, patch_size, n_side):
    """
    patches corner, top, and corner windows into one large array that can me used
    as a weighted (or binary) mask.
    
    :parmam window_type: string
    :param n_side: int, how many patches on the side
    
    :return: nupy array of shape (patch_sixe*n_side, patch_sixe*n_side) 
    """
    implemented_windows = ["hann", "bartley-hann", "triangular", "step"]

    assert (window_type in implemented_windows), "Window function not implemented or misspelled"
    corner_function = corner_hann_window
    center_edge_function = top_hann_window

    n_pix = n_side * patch_size
    out = np.empty((n_pix, n_pix), dtype=np.float32)
    for i in range(n_side):
        for j in range(n_side):
            if (i==0) and (j==0):
                out[0:patch_size, 0:patch_size] = np.rot90(corner_function(patch_size, patch_size),2)
            if (i==0) and (j==(n_side-1)):
                out[0:patch_size, j*patch_size:(j+1)*patch_size] = np.rot90(corner_function(patch_size, patch_size),1)
            if (i==n_side-1) and (j==0):
                out[i*patch_size:(i+1)*patch_size, 0:patch_size] = np.rot90(corner_function(patch_size, patch_size), 3)
            if (i==n_side-1) and (j==n_side-1):
                out[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = np.rot90(corner_function(patch_size, patch_size), 0)

    
    return out

