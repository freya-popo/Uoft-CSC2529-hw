###################################################
#   Release code for CSC2529 HW6, task 3
#
#   Instructions:
#       You don't need to change anything here, please
#       edit the files leastnorm.py, admm_tv.py, and
#       admm_dncnn.py
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/2022
###################################################

# import packages
import numpy as np

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from admm_tv import *
from admm_dncnn import *
from leastnorm import *

# select target image and load it
img = io.imread('uoft_logo.png').astype(float) / 255.
print(img.shape)
# compression factor
compression = 4

N = int(np.round(np.size(img) / compression))
masks = np.random.rand(N, *img.shape)

# image formation model and adjoint
Afun = lambda x: np.sum(masks * x, axis=(-2, -1))  # A
Atfun = lambda x: np.sum(masks * x[:, np.newaxis, np.newaxis], axis=0)  # A.T

# noise parameter - standard deviation
sigma = 0.001

# Simulated measurements
b = Afun(img) + sigma * np.random.randn(N)

# number of iterations for HQS/ADMM
num_iters = 75

# Least-norm solution
x_ln = leastnorm(b, Afun, Atfun, num_iters, img.shape)

# clip results to make sure it's within the range [0,1]
x_ln = np.clip(x_ln, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_LN = round(peak_signal_noise_ratio(img, x_ln), 1)

# ADMM+TV solver
rho = 16  # rho parameter of ADMM
lam = 1.0  # relative weight of TV term
x_admm_tv = admm_tv(b, Afun, Atfun, lam, rho, num_iters, img.shape)
# clip results to make sure it's within the range [0,1]
x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_ADMM_TV = round(peak_signal_noise_ratio(img, x_admm_tv), 1)

# run ADMM+DnCNN
rho = 5.2  # rho parameter of ADMM
lam = 0.05  # relative weight of TV term
x_admm_dncnn = admm_dncnn(b, Afun, Atfun, lam, rho, num_iters, img.shape)
# clip results to make sure it's within the range [0,1]
x_admm_dncnn = np.clip(x_admm_dncnn, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_ADMM_DNCNN = round(peak_signal_noise_ratio(img, x_admm_dncnn), 1)

# show results
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1)
ax.imshow(x_ln, cmap="gray", vmin=0.0, vmax=1.0)
ax.set_title("LN, PSNR: " + str(PSNR_LN))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(1, 3, 2)
ax.imshow(x_admm_tv, cmap="gray", vmin=0.0, vmax=1.0)
ax.set_title("ADMM+TV, PSNR: " + str(PSNR_ADMM_TV))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax = fig.add_subplot(1, 3, 3)
ax.imshow(x_admm_dncnn, cmap="gray", vmin=0.0, vmax=1.0)
ax.set_title("ADMM+DnCNN, PSNR: " + str(PSNR_ADMM_DNCNN))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.tight_layout()
plt.savefig('task3.png')
plt.show()
