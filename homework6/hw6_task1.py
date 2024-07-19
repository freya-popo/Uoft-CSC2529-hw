#########################################################
#   Release code for CSC2529 HW6, task 1
#
#   Instructions:
#       You don't need to change anything here, please
#       edit the file deconv_adam_tv.py
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/22
########################################################

# import packages
import numpy as np
from numpy.fft import fft2, ifft2

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio
from skimage.filters import gaussian

from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

# import our Adam-based deconvolution code
from deconv_adam_tv import *


# helper function for computing a 2D Gaussian convolution kernel
def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0] // 2, size[1] // 2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel / np.sum(kernel)


# select target image and load it
name = 'birds'
img = io.imread(f'{name}.png').astype(float) / 255

# create blur kernel
c = fspecial_gaussian_2d((30, 30), 2.5)

# compute otf of blur kernel
cFT = psf2otf(c, (img.shape[0], img.shape[1]))

# this is our forward image formation model as a function
Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

# standard deviation of sensor noise
sigma = 0.1

# simulated measurements for all 3 color channels
b = np.zeros(np.shape(img))
for it in range(3):
    b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])

# solver parameters
lam = 0.05  # relative weight of TV term
num_iters = 75  # number of iterations for Adam
learning_rate = 5e-2  # learning rate

# run PyTorch-based Adam solver for each color channel with anisotropic TV regularizer
x_adam_tv_anisotropic = np.zeros(np.shape(b))
for it in range(3):
    x_adam_tv_anisotropic[:, :, it] = deconv_adam_tv(b[:, :, it], c, lam, num_iters, learning_rate, True)
# clip results to make sure it's within the range [0,1]
x_adam_tv_anisotropic = np.clip(x_adam_tv_anisotropic, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_ADAM_TV_ANISOTROPIC = round(peak_signal_noise_ratio(img, x_adam_tv_anisotropic), 1)
print(round(peak_signal_noise_ratio(img, x_adam_tv_anisotropic), 3))
# run PyTorch-based Adam solver for each color channel with isotropic TV regularizer
x_adam_tv_isotropic = np.zeros(np.shape(b))
for it in range(3):
    x_adam_tv_isotropic[:, :, it] = deconv_adam_tv(b[:, :, it], c, lam, num_iters, learning_rate, False)
# clip results to make sure it's within the range [0,1]
x_adam_tv_isotropic = np.clip(x_adam_tv_isotropic, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_ADAM_TV_ISOTROPIC = round(peak_signal_noise_ratio(img, x_adam_tv_isotropic), 1)
print(round(peak_signal_noise_ratio(img, x_adam_tv_isotropic), 3))

# show results
fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
ax.imshow(img)
ax.set_title("Target Image", fontsize=10)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 2)
ax.imshow(b)
ax.set_title("Blurry and Noisy Image", fontsize=10)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 3)
ax.imshow(x_adam_tv_anisotropic)
ax.set_title("Adam + Anisotropic TV, PSNR: " + str(PSNR_ADAM_TV_ANISOTROPIC), fontsize=10)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(2, 2, 4)
ax.imshow(x_adam_tv_isotropic)
ax.set_title("Adam + Isotropic TV, PSNR: " + str(PSNR_ADAM_TV_ISOTROPIC), fontsize=10)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig('task1.png')
plt.show()
