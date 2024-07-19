from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from pdb import set_trace

from bilateral import bilateral2d
from fspecial import fspecial_gaussian_2d
from nlm import nonlocalmeans

clean = io.imread('night.png').astype(float) / 255
noisy = io.imread('night_downsampled_noisy_sigma_0.0781.png').astype(float) / 255

# Store outputs in dictionary
filtered = {}

# Choose standard deviations:
sigmas = [0.5, 1, 2]
# sigmas = [1, 2, 3]
# fig1, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
# fig2, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
for sigma in sigmas:
    filtSize = 2 * sigma + 1

    # Gaussian filter
    out1 = np.zeros_like(noisy)
    for channel in [0, 1, 2]:
        out1 = gaussian(noisy, sigma=sigma, channel_axis=channel)
    filtered[f'gaussian_{sigma}'] = out1
    # ax[sigma-1].imshow(out1)
    # ax[sigma-1].set_title(f'Gaussian (sigma={sigma})')
    # ax[sigma-1].axis('off')

    # Median filter
    out2 = np.zeros_like(noisy)
    for channel in [0, 1, 2]:
        out2[:, :, channel] = median_filter(noisy[:, :, channel], size=int(filtSize))
    filtered[f'median_{filtSize}'] = out2
    # ax[sigma-1].imshow(out2)
    # ax[sigma-1].set_title(f'Median (sigma={sigma})')
    # ax[sigma-1].axis('off')

    # Bilateral Filter
    for sigmaIntensity in [0.25, 0.5]:
        # sigmaIntensity = 0.25
        bilateral = np.zeros_like(noisy)
        for channel in [0, 1, 2]:
            bilateral[..., channel] = bilateral2d(noisy[..., channel],
                                                  radius=int(sigma),
                                                  sigma=sigma,
                                                  sigmaIntensity=sigmaIntensity)
        if sigmaIntensity == 0.25:
            filtered[f'bilateral_{sigma}_sigmaIntensity=0.25'] = bilateral
        else:
            filtered[f'bilateral_{sigma}_sigmaIntensity=0.5'] = bilateral

    # Non-local means
    # May take some time - crop the noisy image if you want to debug faster
    nlmSigma = 0.1  # Feel free to modify
    searchWindowRadius = 5
    averageFilterRadius = int(sigma)
    nlm = np.zeros_like(noisy)
    print('yesss')
    for channel in [0, 1, 2]:
        nlm[..., channel:channel + 1] = nonlocalmeans(noisy[..., channel:channel + 1],
                                                      searchWindowRadius,
                                                      averageFilterRadius,
                                                      sigma,
                                                      nlmSigma)
    filtered[f'nlm_{sigma}'] = nlm

# plt.tight_layout()
# plt.savefig('task3-guassian-123.png')
# plt.savefig('task3-median-123.png')
# plt.show()

# Sample plotting code, feel free to modify!
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(50, 40))
ax[0, 0].imshow(clean)
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')
ax[0, 1].imshow(noisy)
ax[0, 1].set_title('Noisy')
ax[0, 1].axis('off')
for r, sigma in enumerate(sigmas):
    # Plot Gaussian
    ax[r + 1, 0].imshow(filtered[f'gaussian_{sigma}'])
    ax[r + 1, 0].set_title(f'Gaussian (sigma={sigma})')
    ax[r + 1, 0].axis('off')

    # Plot Median
    filtSize = 2 * sigma + 1
    ax[r + 1, 1].imshow(filtered[f'median_{filtSize}'])
    ax[r + 1, 1].set_title(f'Median (filter size = {filtSize})')
    ax[r + 1, 1].axis('off')

    # Plot Bilateral1
    ax[r + 1, 2].imshow(filtered[f'bilateral_{sigma}_sigmaIntensity=0.25'])
    ax[r + 1, 2].set_title(f'Bilateral filter (sigmaSpatial={sigma}), sigmaIntensity=0.25')
    ax[r + 1, 2].axis('off')

    # Plot Bilateral1
    ax[r + 1, 3].imshow(filtered[f'bilateral_{sigma}_sigmaIntensity=0.5'])
    ax[r + 1, 3].set_title(f'Bilateral filter (sigmaS={sigma}), sigmaI=0.5')
    ax[r + 1, 3].axis('off')

    # Plot Non-local means
    ax[r + 1, 4].imshow(filtered[f'nlm_{sigma}'])
    ax[r + 1, 4].set_title(
        f'Non-local means (sigmaS={sigma}, sigmaNlm={nlmSigma}, searchRadius={searchWindowRadius})')
    ax[r + 1, 4].axis('off')
ax[0, 2].remove()
ax[0, 3].remove()
ax[0, 4].remove()
fig.savefig('task3_denoising_1.png', bbox_inches='tight')
