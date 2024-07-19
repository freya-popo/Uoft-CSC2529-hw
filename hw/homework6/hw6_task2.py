import numpy as np
from numpy.fft import fft2, ifft2

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.filters import gaussian

from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

from deconv_admm_tv import *
from deconv_admm_dncnn import *


def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)


# Select image
name = 'birds'
img = io.imread(f'{name}.png').astype(float)/255

# blur kernel
c = fspecial_gaussian_2d((30, 30), 2.5)

# Blur kernel
cFT = psf2otf(c, (img.shape[0], img.shape[1]))
Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

# noise parameter - standard deviation
sigma = 0.1

# simulated measurements
b = np.zeros(np.shape(img))
for it in range(3):
    b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])

# ADMM parameters for TV prior
num_iters = 75
rho = 5
lam = 0.025

# run ADMM+TV solver
x_admm_tv = np.zeros(np.shape(b))
for it in range(3):
    x_admm_tv[:, :, it] = deconv_admm_tv(b[:, :, it], c, lam, rho, num_iters)
x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
PSNR_ADMM_TV = round(compute_psnr(img, x_admm_tv), 1)

# ADMM parameters for DnCNN prior
num_iters = 75
lam = 0.01 * 0.5
rho = 1 * 0.5

# run ADMM+DnCNN solver
x_admm_dncnn = np.zeros(np.shape(b))
for it in range(3):
    x_admm_dncnn[:, :, it] = deconv_admm_dncnn(b[:, :, it], c, lam, rho, num_iters)
x_admm_dncnn = np.clip(x_admm_dncnn, 0.0, 1.0)
PSNR_ADMM_DNCNN = round(compute_psnr(img, x_admm_dncnn), 1)

# show image
fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
ax.imshow(img)
ax.set_title("Target Image")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 2)
ax.imshow(b)
ax.set_title("Blurry and Noisy Image")
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 3)
ax.imshow(x_admm_tv)
ax.set_title("ADMM TV, PSNR: " + str(PSNR_ADMM_TV))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 4)
ax.imshow(x_admm_dncnn)
ax.set_title("ADMM DnCNN, PSNR: " + str(PSNR_ADMM_DNCNN))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig('task2.png')
plt.show()
