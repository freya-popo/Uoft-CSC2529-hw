import numpy as np
from fspecial import fspecial_gaussian_2d


def inbounds(img, y, x):
    return 0 <= y and y < img.shape[0] and \
           0 <= x and x < img.shape[1]


# w(i,j)
def comparePatches(patch1, patch2, kernel, sigma):
    return np.exp(-np.sum(kernel * (patch1 - patch2) ** 2) / (2 * sigma ** 2))


def nonlocalmeans(img, searchWindowRadius, averageFilterRadius, sigma, nlmSigma):
    # averagefilterradius -- small filter radius/ patch radius
    # searchwindowradius -- neighbour radius
    # Initialize output to 0
    out = np.zeros_like(img)
    # Pad image to reduce boundary artifacts
    pad = max(averageFilterRadius, searchWindowRadius)
    print('pad', pad)
    imgPad = np.pad(img, pad)
    imgPad = imgPad[..., pad:-pad]  # Don't pad third channel

    # Smoothing kernel
    filtSize = (2 * averageFilterRadius + 1, 2 * averageFilterRadius + 1)
    kernel = fspecial_gaussian_2d(filtSize, sigma)
    # Add third axis for broadcasting
    kernel = kernel[:, :, np.newaxis]
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerPatch = imgPad[y + pad - averageFilterRadius:y + pad + averageFilterRadius + 1,
                          x + pad - averageFilterRadius:x + pad + averageFilterRadius + 1,
                          :]
            # print(y + pad - averageFilterRadius, y + pad + averageFilterRadius + 1)
            # print(y + pad - searchWindowRadius, y + pad + searchWindowRadius + 1)
            # print(comparePatches(centerPatch,centerPatch,kernel,sigma))
            # Go over a window around the current pixel, compute weights
            # based on difference of patches, sum the weighted intensity
            # Hint: Do NOT include the patches centered at the current pixel
            # in this loop, it will throw off the weights
            weights = np.zeros((2 * searchWindowRadius + 1, 2 * searchWindowRadius + 1, 1))
            # print(weights)
            weight_all = 0
            max_weight = 0
            for i in range(y + pad - searchWindowRadius, y + pad + searchWindowRadius + 1):
                for j in range(x + pad - searchWindowRadius, x + pad + searchWindowRadius + 1):
                    if inbounds(img, i - pad, j - pad):
                        patch = imgPad[i - averageFilterRadius:i + averageFilterRadius + 1,
                                j - averageFilterRadius:j + averageFilterRadius + 1,
                                :]
                        # print('aaa', i, j, y, x)
                        # print('bbb', i - averageFilterRadius, i + averageFilterRadius + 1)
                        # print('bbb', j - averageFilterRadius, j + averageFilterRadius + 1)
                        weight = comparePatches(patch, centerPatch, kernel, nlmSigma)
                        # if i == y + pad and j == x + pad:
                        #     print('i=y', i, j, weight)
                        #     # print(i - (y + pad - searchWindowRadius), j - (x + pad - searchWindowRadius))
                        # # if i != y and j != x:
                        # else:
                        weights[i - (y + pad - searchWindowRadius)][j - (x + pad - searchWindowRadius)] = weight
                        # print(weight)
                        weight_all += weight
                        max_weight = max(weight, max_weight)
                        # print(weight)
            # print('with pad', searchWindowRadius,searchWindowRadius)
            weights[searchWindowRadius, searchWindowRadius] = max_weight
            weight_all += max_weight
            # print(weights, weight_all, max_weight)
            weights = weights / weight_all
            # print(np.sum(weights))
            # print('___________________________')
            # This makes it a bit better: Add current pixel as well with max weight
            # computed from all other neighborhoods.
            v = 0
            for i in range(y + pad - searchWindowRadius, y + pad + searchWindowRadius + 1):
                for j in range(x + pad - searchWindowRadius, x + pad + searchWindowRadius + 1):
                    v += imgPad[i][j] * weights[i - (y + pad - searchWindowRadius)][j - (x + pad - searchWindowRadius)]
            out[y, x, :] = v  # TODO: Replace with your code.
    return out
