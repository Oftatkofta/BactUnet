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


def corner_step_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a step window
    """
    out = np.zeros((height, width), np.float32)
    out[0:int(height*0.75),0:int(width*0.75)] = 1
    return out

def top_step_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a step window
    """
    out = np.zeros((height, width), np.float32)
    out[0:int(height*0.75),int(width*0.25):int(width*0.75)]=1
    return out

def center_step_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a step window
    """
    out = np.zeros((height, width), np.float32)
    out[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]=1
    return out


def corner_triangular_window(height, width):
    """
    :param i: height in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a triangular window
    """
    out = np.zeros((height, width), np.float32)
    I = 2/height
    J = 2/width

    for i in range(height):
        for j in range(width):
            if (i <= height/2) and (j >= width/2):
                out[i, j] =  (1 - abs(J*j - 1))
            elif (i <= height/2) and (j < width/2):
                out[i, j] = 1
            elif (i > height/2) and (j < width/2):
                out[i, j] = (1 - abs(I * i - 1))

            else:
                out[i,j] = (1-abs(I*i -1)) * (1 - abs(J*j - 1))

    return out

def top_triangular_window(height, width):
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
            if (i <= height/2):
                out[i, j] =  (1 - abs(J*j - 1))
            else:
                out[i,j] = (1-abs(I*i -1)) * (1 - abs(J*j - 1))

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

    type_dict = {
        'hann': [corner_hann_window, top_hann_window, hann_window],
        'step': [corner_step_window, top_step_window, center_step_window],
        'triangular': [corner_triangular_window, top_triangular_window, triangular_window]
    }
    assert (window_type in type_dict.keys()), "Window function not implemented or misspelled"


    corner_function = type_dict[window_type][0]
    edge_function = type_dict[window_type][1]
    center_function = type_dict[window_type][2]

    n_pix = n_side * patch_size
    out = np.empty((n_pix, n_pix), dtype=np.float32)
    for i in range(n_side):
        for j in range(n_side):
            if (i==0):
                if (j==0):
                    out[0:patch_size, 0:patch_size] = np.rot90(corner_function(patch_size, patch_size),0)
                elif  (j==(n_side-1)):
                    out[0:patch_size, j*patch_size:(j+1)*patch_size] = np.rot90(corner_function(patch_size, patch_size),3)
                else:
                    out[0:patch_size, j * patch_size:(j + 1) * patch_size] = np.rot90(
                        edge_function(patch_size, patch_size), 0)

            elif (i==n_side-1):
                if (j==0):
                    out[i*patch_size:(i+1)*patch_size, 0:patch_size] = np.rot90(corner_function(patch_size, patch_size), 1)
                elif (j==n_side-1):
                    out[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = np.rot90(corner_function(patch_size, patch_size), 2)
                else:
                    out[i*patch_size:(i+1)*patch_size, j * patch_size:(j + 1) * patch_size] = np.rot90(
                        edge_function(patch_size, patch_size), 2)
            elif (j==0):
                out[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = np.rot90(
                    edge_function(patch_size, patch_size), 1)
            elif (j==(n_side-1)):
                out[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = np.rot90(
                    edge_function(patch_size, patch_size), 3)
            else:
                out[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = center_function(patch_size, patch_size)

    
    return out

