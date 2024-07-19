import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import imageio
import cv2
from pdb import set_trace

# initialize HDR image with all zeros
hdr = np.zeros((768, 512, 3), dtype=float)
scale = np.zeros((768, 512, 3), dtype=float)
T = []
with open('hdr_data/memorial.hdr_image_list.txt', 'r') as file:
    i = 0
    for line in file:
        i += 1
        if i > 3:
            T.append(1 / float(line.split(' ')[1]))


##########################################
def gamma(img):
    return img ** 2.2


def weight(img):
    return np.exp(-4 * (img - 0.5) ** 2 / 0.5 ** 2)


# load LDR images from hdr_dir (need to unzip before)
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(30, 15))
for i in range(61, 77):
    img = io.imread('hdr_data/memorial00' + str(i) + '.png').astype(float) / 255
    weight_i = weight(gamma(img))
    hdr += weight_i * (np.log(gamma(img) + np.finfo(np.float32).eps) - np.log(T[i - 61]))
    scale += weight_i
    axes[(i - 61) // 8, (i - 61) % 8].imshow(np.clip(weight_i, a_min=0, a_max=1))
    # axes[(i-61)//8, i].set_title(f'blurred image with sigma={}', fontsize=30)
    axes[(i - 61) // 8, (i - 61) % 8].axis('off')
plt.tight_layout()
# plt.savefig("hw4-task1-weights.png")
plt.show()
# fuse LDR images using weights, make sure to store your fused HDR using the name hdr
# hdr = ...
##########################################

# Normalize
hdr = np.exp(hdr / scale)
hdr *= 0.8371896 / np.mean(
    hdr)  # this makes the mean of the created HDR image match the reference image (totally optional)

# convert to 32 bit floating point format, required for OpenCV
hdr = np.float32(hdr)

# crop boundary - image data here are only captured in some of the exposures, which is why they are indicated in blue in the LDR images
hdr = hdr[29:720, 19:480, :]
s = [0.1, 0.5, 1]
y = [5, 2.2, 1]
hdr0 = hdr
fig1, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 20))
for i in range(3):
    for j in range(3):
        hdr = (s[i] * hdr0) ** (float(1 / y[j]))
        ax[i, j].imshow(np.clip(hdr, a_min=0, a_max=1))
        ax[i, j].set_title(f'scale={s[i]},gamma=1/{y[j]}', fontsize=15)
        ax[i, j].axis('off')
plt.tight_layout()
plt.savefig("hw4-task1-s_y.png")
plt.show()

# tonemap image and save LDR image using OpenCV's implementation of Drago's tonemapping operator
gamma = 1.0
saturation = 0.7
bias = 0.85
tonemapDrago = cv2.createTonemapDrago(gamma, saturation, bias)
ldrDrago = tonemapDrago.process(hdr0)
io.imsave('my_hdr_image_tonemapped.jpg', np.uint8(np.clip(3 * ldrDrago, 0, 1) * 255))

# write HDR image (can compare to hw4_1_memorial_church.hdr reference image in an external viewer)
hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
cv2.imwrite('my_hdr_image.hdr', hdr)
