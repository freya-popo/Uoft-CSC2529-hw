import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import skimage.io as io
from scipy.signal import convolve2d
from pypher.pypher import psf2otf
import matplotlib.pyplot as plt
import time
from fspecial import fspecial_gaussian_2d

img = io.imread('birds_gray.png').astype(float) / 255
img = img[:, :, 0]
spatial_times = []
fourier_times = []

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
sigmas = [0.1, 1, 10]
for k in range(len(sigmas)):
    sigma = sigmas[k]
    filtSize = np.ceil(9 * sigma).astype(int)
    lp = fspecial_gaussian_2d((filtSize, filtSize), sigma)

    ### Your code here ###
    spatial_start = time.time()
    spatial_result = convolve2d(img, lp, mode='valid', boundary='wrap')
    spatial_times.append(time.time() - spatial_start)
    axs[0, k].imshow(np.clip(spatial_result, a_min=0, a_max=1), cmap='gray')
    axs[0, k].set_title(f'spatial filter (sigma={sigma})', fontsize=30)
    axs[0, k].axis('off')

    fourier_start = time.time()
    img_ft = fft2(img)
    img_ftshift = fftshift(img_ft)
    ft = psf2otf(lp, img.shape)
    ftshift = fftshift(ft)
    ft_mid = img_ftshift * ftshift
    ishift = ifftshift(ft_mid)
    ft_result = ifft2(ishift).real

    fourier_times.append(time.time() - fourier_start)
    axs[1, k].imshow(np.clip(ft_result, a_min=0, a_max=1.), cmap='gray')
    axs[1, k].set_title(f'fourier filter (sigma={sigma})', fontsize=30)

    axs[1, k].axis('off')

    axs[2, k].imshow(ftshift, cmap='gray')
    axs[2, k].set_title(f'OFT(sigma={sigma})', fontsize=30)
    axs[2, k].axis('off')
fig.savefig('task1_a_filter.png', bbox_inches='tight')
plt.show()

fig1, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].bar(sigmas, spatial_times, width=0.4, label='Spatial Convolution')
for i, j in zip(sigmas, spatial_times):
    axs[0].text(i, j, round(j, 3), ha='center', va='bottom')
axs[0].set_title(f'spatial filter')
axs[0].set_xlabel('different sigmas')
axs[0].set_ylabel('Time (s)')

axs[1].bar(sigmas, fourier_times, width=0.4, label='Fourier Convolution')
for i, j in zip(sigmas, fourier_times):
    axs[1].text(i, j, round(j, 3), ha='center', va='bottom')
axs[1].set_title(f'fourier filter')

axs[1].set_xlabel('different sigmas')
axs[1].set_ylabel('Time (s)')
axs[0].set_yscale('log')
fig1.savefig('task1_a_time.png', bbox_inches='tight')
plt.show()
