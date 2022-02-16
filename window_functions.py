import numpy as np
import math
from matplotlib import pyplot as plt
import tifffile as tiff

def hann_window(height, width):
    """
    https://doi.org/10.1371/journal.pone.0229839
    :param i: width in pixels
    :param j: height in pixels
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
    :param i: width in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a upper left corner Hann window
    """
    out = np.empty((height, width), np.float32)
    I = 2*math.pi/(height-1)
    J = 2*math.pi/(width-1)
    for i in range(height):
        for j in range(width):
            if (i <= width/2) and (j <= height/2):
                out[i, j] = 1
            elif (i > width/2) and (j < height/2):
                out[i, j] = 0.5 * (1 - math.cos(I * i))
            elif (i < width/2) and (j > height/2):
                out[i,j] = 0.5 * (1 - math.cos(J*j))
            else:
                out[i,j] = 0.25 * (1-math.cos(I*i)) * (1-math.cos(J*j))

    return out

def top_hann_window(height, width):
    """
    :param i: width in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a upper edge Hann window
    """
    out = np.empty((height, width), np.float32)
    I = 2*math.pi/(height-1)
    J = 2*math.pi/(width-1)
    for i in range(height):
        for j in range(width):
            if (j < width/2):
                out[i, j] = 0.5 * (1 - math.cos(I * i))
            else:
                out[i,j] = 0.25 * (1-math.cos(I*i)) * (1-math.cos(J*j))

    return out

def bartley_hann_window(width, height):
    """
    :param i: width in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a Bartley-Hann window
    """
    out = np.empty((width, height), np.float32)

    a0 = 0.62
    a1 = 0.48
    a2 = 0.38

    I = 2*math.pi/(width)
    J = 2*math.pi/(height)

    for i in range(width):
        for j in range(height):
            out[i,j] = (a0 + a1 * abs((i / width) - 0.5) - a2 * math.cos(I * i)) * (a0 + a1 * abs((j / height) - 0.5) - a2 * math.cos(J * j))

    return out

def triangular_window(width, height):
    """
    :param i: width in pixels
    :param j: height in pixels
    :return: nupy array of shape (i, j) of a triangular window
    """
    out = np.empty((width, height), np.float32)
    I = 2/width
    J = 2/height

    for i in range(width):
        for j in range(height):
            out[i,j] = (1-abs(I*i -1)) * (1 - abs(J*j - 1))

    return out


indir = r"C:\Users\Jens\Documents\Code\bactnet\Bactnet\Training data\stacks\predict\piccolo"

plt.imshow(top_hann_window(450, 900))
plt.show()