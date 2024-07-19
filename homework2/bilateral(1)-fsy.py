import numpy as np
from fspecial import fspecial_gaussian_2d

def bilateral2d(img, radius, sigma, sigmaIntensity):
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.
            window = imgPad[y:y+2*radius+1, x:x+2*radius+1]
            intensityKernel = np.exp(-np.power(window - centerVal, 2) / (2 * sigmaIntensity ** 2))
            kernel = np.multiply(spatialKernel, intensityKernel)
            Wp = np.sum(kernel)
            out[y, x] = np.sum(np.multiply(window,  kernel)) / Wp # TODO: Replace with your own code
    return out
