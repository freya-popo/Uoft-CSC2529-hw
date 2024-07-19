import numpy as np
import skimage.color
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp2d
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
from scipy import signal

np.set_printoptions(threshold=sys.maxsize)

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir / 'lighthouse_RAW_noisy_sigma0.01.png')
raw = img1.astype(np.float64) / 255
img_original = io.imread(hw_dir / 'lighthouse.png')

####(a)######
R = np.zeros(raw.shape, np.float64)
G = np.zeros(raw.shape, np.float64)
B = np.zeros(raw.shape, np.float64)
x_red = []
y_red = []
z_red = []
x_blue = []
y_blue = []
z_blue = []
x = np.arange(0, raw.shape[1], 1)
y = np.arange(0, raw.shape[0], 1)
for i in range(raw.shape[0]):
    z_red_m = []
    z_blue_m = []
    for j in range(raw.shape[1]):
        if i % 2 == 0 and j % 2 == 0:  # red
            R[i][j] = raw[i][j]
            if i not in y_red:
                y_red.append(i)
            if j not in x_red:
                x_red.append(j)
            z_red_m.append(raw[i][j])
        elif i % 2 == 1 and j % 2 == 1:  # blue
            B[i][j] = raw[i][j]
            if i not in y_blue:
                y_blue.append(i)
            if j not in x_blue:
                x_blue.append(j)
            z_blue_m.append(raw[i][j])
        else:
            G[i][j] = raw[i][j]
    if len(z_red_m) != 0:
        z_red.append(z_red_m)
    if len(z_blue_m) != 0:
        z_blue.append(z_blue_m)

f_red = interp2d(x_red, y_red, z_red, kind='linear')
f_blue = interp2d(x_blue, y_blue, z_blue, kind='linear')
R_final = f_red(x, y)
B_final = f_blue(x, y)
G_up = np.roll(G, 1, axis=0)
G_down = np.roll(G, -1, axis=0)
G_left = np.roll(G, -1, axis=1)
G_right = np.roll(G, 1, axis=1)
G_final = (G_up + G_down + G_left + G_right) / 4 + G

img_2 = np.stack([R_final, G_final, B_final], axis=2)
img_2_gamma = img_2 ** (1 / 2.2)

psnr1 = psnr(img_original, img_2_gamma * 255)
# psnr2 = psnr(img_original.astype(np.float64) / 255, img_2_gamma)
print('psnr for linear demosaicing is', psnr1)
# print(psnr2)
# fig1 = plt.imshow(np.clip(R_final * 255, a_min=0, a_max=255.).astype(np.uint8))
# plt.show()
# fig2 = plt.imshow(np.clip(B_final * 255, a_min=0, a_max=255.).astype(np.uint8))
# plt.show()
# fig3 = plt.imshow(np.clip(G_final * 255, a_min=0, a_max=255.).astype(np.uint8))
# plt.show()
#
fig1 = plt.imshow(np.clip(img_2_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
# io.imsave("task2_a.png", np.clip(img_2_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
plt.show()

####(b)######
filter_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
               29, 30]
psnr2_lst = []
fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(50, 60))
for s in filter_size:

    img_3 = skimage.color.rgb2ycbcr(img_2)
    cb = ndimage.median_filter(img_3[:, :, 1], size=s)
    cr = ndimage.median_filter(img_3[:, :, 2], size=s)
    img_5 = np.zeros_like(img_3)
    img_5[:, :, 0] = img_3[:, :, 0]
    img_5[:, :, 1] = cb
    img_5[:, :, 2] = cr
    img_6 = skimage.color.ycbcr2rgb(img_5)
    img_7 = np.clip(img_6, 0, 1)
    img_7_gamma = img_7 ** (1 / 2.2)
    psnr2 = psnr(img_original.astype(np.float64) / 255, img_7_gamma)
    psnr2_lst.append(psnr2)
    print('when filter size is', s, 'psnr for linear+smoothing is', psnr2)
    ax[int((s-1) / 5), (s-1) % 5].imshow(np.clip(img_7_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
    ax[int((s-1) / 5), (s-1) % 5].set_title(f'filter size={s}')
    ax[int((s-1) / 5), (s-1) % 5].axis('off')
plt.tight_layout()
plt.savefig('task2_b_different_size.png')
plt.show()

plt.plot(filter_size, psnr2_lst)
plt.xlabel("filter size")
plt.ylabel("psnr")
plt.savefig("task2_b_psnr.png")
plt.show()
# fig2 = plt.imshow(np.clip(img_7_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
# io.imsave("task2_b.png", np.clip(img_7_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
# plt.show()


####(c)######
# calculate green based on red/blue pixels
filter1 = [[0, 0, -1, 0, 0],
           [0, 0, 2, 0, 0],
           [-1, 2, 4, 2, -1],
           [0, 0, 2, 0, 0],
           [0, 0, -1, 0, 0]]
# calculate red/blue(r/b row, b/r column) based on green
filter2 = [[0, 0, 1 / 2, 0, 0],
           [0, -1, 0, -1, 0],
           [-1, 4, 5, 4, -1],
           [0, -1, 0, -1, 0],
           [0, 0, 1 / 2, 0, 0]]
# calculate red/blue(b/r row, r/b column) based on green
filter3 = [[0, 0, -1, 0, 0],
           [0, -1, 4, -1, 0],
           [1 / 2, 0, 5, 0, 1 / 2],
           [0, -1, 4, -1, 0],
           [0, 0, -1, 0, 0]]
# calculate red/blue based on blue/red
filter4 = [[0, 0, -3 / 2, 0, 0],
           [0, 2, 0, 2, 0],
           [-3 / 2, 0, 6, 0, -3 / 2],
           [0, 2, 0, 2, 0],
           [0, 0, -3 / 2, 0, 0]]

m1 = signal.convolve2d(raw, filter1, boundary='symm', mode='same') / 8
m2 = signal.convolve2d(raw, filter2, boundary='symm', mode='same') / 8
m3 = signal.convolve2d(raw, filter3, boundary='symm', mode='same') / 8
m4 = signal.convolve2d(raw, filter4, boundary='symm', mode='same') / 8

for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        if i % 2 == 0 and j % 2 == 0:  # red point
            G[i][j] = m1[i][j]
            B[i][j] = m4[i][j]
        elif i % 2 == 1 and j % 2 == 1:  # blue point
            G[i][j] = m1[i][j]
            R[i][j] = m4[i][j]
        else:
            if i % 2 == 0 and j % 2 == 1:
                R[i][j] = m2[i][j]
                B[i][j] = m3[i][j]
            elif i % 2 == 1 and j % 1 == 0:
                B[i][j] = m2[i][j]
                R[i][j] = m3[i][j]

img_8 = np.stack([R, G, B], axis=2)
img_9 = np.clip(img_8, 0, 1)
img_9_gamma = img_9 ** (1 / 2.2)
psnr3 = psnr(img_original.astype(np.float64) / 255, img_9_gamma)
print('psnr for high quality', psnr3)

fig3 = plt.imshow(np.clip(img_9_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
# io.imsave("task2_c.png", np.clip(img_9_gamma * 255, a_min=0, a_max=255.).astype(np.uint8))
plt.show()
