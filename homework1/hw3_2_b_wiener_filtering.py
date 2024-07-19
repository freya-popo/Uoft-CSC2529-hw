import math

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from pypher.pypher import psf2otf
from skimage.metrics import peak_signal_noise_ratio as psnr
from pdb import set_trace
from fspecial import fspecial_gaussian_2d

img = io.imread('birds_gray.png', as_gray=True).astype(float) / 255

# Task 2b - Wiener filtering

c = fspecial_gaussian_2d((35, 35), 5.)
cFT = psf2otf(c, img.shape)
cFT_shift = fftshift(cFT)

# Blur image with kernel
blur = convolve2d(img, c, mode='same', boundary='wrap')

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(40, 20))
sigmas = [0, 0.001, 0.01, 0.1]
for i in range(4):
    sigma = sigmas[i]
    # Add noise to blurred image
    unfilt = blur + sigma * np.random.randn(*blur.shape)

    ### Your code here ###
    axs[0, i].imshow(np.clip(unfilt, a_min=0, a_max=1), cmap='gray')
    axs[0, i].set_title(f'blurred image with sigma={sigma}', fontsize=30)
    axs[0, i].axis('off')
    if sigma == 0:
        SNR = math.inf
    else:
        SNR = np.mean(unfilt) / sigma

    img_ft = fft2(unfilt)
    img_ftshift = fftshift(img_ft)
    H = (1 / cFT_shift) * ((abs(cFT_shift) ** 2) / (abs(cFT_shift) ** 2 + 1 / SNR))
    ft_mid = img_ftshift * H
    ishift = ifftshift(ft_mid)
    ft_result = ifft2(ishift).real
    axs[1, i].imshow(np.clip(ft_result, a_min=0, a_max=1), cmap='gray')
    axs[1, i].set_title(f'processed image with sigma={sigma}', fontsize=30)
    axs[1, i].axis('off')
    PSNR = psnr(img, ft_result)
    print('when the sigma of noise is ', sigma, ', the psnr is ', PSNR)
plt.savefig('task2_b.png', bbox_inches='tight')
plt.show()
