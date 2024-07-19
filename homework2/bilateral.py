import math

import numpy as np
from fspecial import fspecial_gaussian_2d
import math


def bilateral2d(img, radius, sigma, sigmaIntensity):
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2 * radius + 1, 2 * radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    # print(spatialKernel)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            c_y = y + pad
            c_x = x + pad
            # print(c_y, c_x)
            centerVal = imgPad[c_y, c_x]  # Careful of padding amount!  # pad=radius
            # print('center', c_y, c_x, centerVal)
            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.
            W = 0
            F = 0
            # print(imgPad[c_y - radius: c_y + radius + 1, c_x - radius: c_x + radius + 1])
            for i in range(c_y - radius, c_y + radius + 1):
                for j in range(c_x - radius, c_x + radius + 1):
                    I_x_i = imgPad[i][j]
                    # print(I_x_i)
                    f = math.exp(-(I_x_i - centerVal) ** 2 / (2 * sigmaIntensity ** 2))
                    # print(f)
                    g = spatialKernel[i + radius - c_y, j + radius - c_x]
                    # print(g)
                    # print(i + radius - c_y, j + radius - c_x)
                    W += f * g
                    F += I_x_i * f * g
                    # print(F)
            # print(W, 'F', F, F / W)
            out[y, x] = F / W
    return out
