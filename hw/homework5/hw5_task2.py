import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

# set random seeds
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BSDS300Dataset(Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True):
        files = sorted(glob(os.path.join(root, 'images', split, '*')))

        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def load_images(self, files):
        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.
            out.append(torch.from_numpy(img))
        return torch.stack(out)

    def patchify(self, img_array, patch_size):
        # create patches from image array of size (N_images, 3, rows, cols)
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        if self.use_patches:
            return self.patches.shape[0]
        else:
            return self.images.shape[0]

    def __getitem__(self, idx):
        if self.use_patches:
            return self.patches[idx]
        else:
            return self.images[idx]


class DnCNN(nn.Module):
    """
    Network architecture from this reference. Note that we omit batch norm
    since we are using a shallow network to speed up training times.

    @article{zhang2017beyond,
      title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
      author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
      journal={IEEE Transactions on Image Processing},
      year={2017},
      volume={26},
      number={7},
      pages={3142-3155},
    }
    """

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=3, use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same', bias=use_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def add_noise(x, sigma=0.1):
    return x + torch.randn_like(x) * sigma


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt) ** 2).mean().item())
    return out


def plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                 val_losses, val_psnrs, val_iters, train_dataset,
                 val_dataset, val_dataloader):
    with torch.no_grad():
        model.eval()

        # evaluate on training dataset sample
        train_dataset.use_patches = False
        train_image = train_dataset[0][None, ...].to(device)
        train_dataset.use_patches = True

        noisy_train_image = add_noise(train_image, sigma=sigma)
        denoised_train_image = model(noisy_train_image)

        # evaluate on validation dataset sample
        val_dataset.use_patches = False
        val_image = val_dataset[6][None, ...].to(device)
        val_dataset.use_patches = True
        val_patch_samples = next(iter(val_dataloader)).to(device)

        # calculate validation metrics
        noisy_val_patch_samples = add_noise(val_patch_samples, sigma=sigma)
        denoised_val_patch_samples = model(noisy_val_patch_samples)
        val_loss = torch.mean((val_patch_samples - denoised_val_patch_samples) ** 2)
        val_psnr = calc_psnr(denoised_val_patch_samples, val_patch_samples)

        val_losses.append(val_loss.item())
        val_psnrs.append(val_psnr)
        val_iters.append(idx)

        noisy_val_image = add_noise(val_image, sigma=sigma)
        denoised_val_image = model(noisy_val_image)

    plt.clf()
    plt.subplot(241)
    plt.plot(losses, label='Train loss')
    plt.plot(val_iters, val_losses, '.', label='Val. loss')
    plt.yscale('log')
    plt.legend()
    plt.title('loss')

    plt.subplot(245)
    plt.plot(psnrs, label='Train PSNR')
    plt.plot(val_iters, val_psnrs, '.', label='Val. PSNR')
    plt.plot(baseline_psnrs, label='Baseline PSNR')
    plt.ylim((0, 32))
    plt.legend()
    plt.title('psnr')

    plt.subplot(242)
    plt.imshow(img_to_numpy(train_image))
    plt.ylabel('Training Set')
    plt.title('GT')

    plt.subplot(243)
    plt.imshow(img_to_numpy(noisy_train_image))
    plt.title('Noisy Image')

    plt.subplot(244)
    plt.imshow(img_to_numpy(denoised_train_image))
    plt.title('Denoised Image')

    plt.subplot(246)
    plt.imshow(img_to_numpy(val_image))
    plt.ylabel('Validation Set')
    plt.title('GT')

    plt.subplot(247)
    plt.imshow(img_to_numpy(noisy_val_image))
    plt.title('Noisy Image')

    plt.subplot(248)
    plt.imshow(img_to_numpy(denoised_val_image))
    plt.title('Denoised Image')

    plt.tight_layout()
    plt.pause(0.1)


def train(sigma=0.1, use_bias=True, hidden_channels=32, epochs=2, batch_size=32, plot_every=200):
    print(f'==> Training on noise level {sigma:.02f} | use_bias: {use_bias} | hidden_channels: {hidden_channels}')

    # create datasets
    train_dataset = BSDS300Dataset(patch_size=32, split='train', use_patches=True)
    val_dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=True)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DnCNN(use_bias=use_bias, hidden_channels=hidden_channels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psnrs = []
    baseline_psnrs = []
    val_losses = []
    val_psnrs = []
    val_iters = []
    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample in train_dataloader:

            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            loss = torch.mean((denoised_sample - sample) ** 2)
            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # plot results
            if not idx % plot_every:
                plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                             val_losses, val_psnrs, val_iters, train_dataset,
                             val_dataset, val_dataloader)

            idx += 1
            pbar.update(1)

    pbar.close()
    return model


def evaluate_model(model, sigma=0.1, output_filename='out.png'):
    dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=False)
    model.eval()

    psnrs = []
    for idx, image in enumerate(dataset):
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_noise(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)

        # include the tiger image in your homework writeup
        if idx == 6:
            skimage.io.imsave(output_filename, (img_to_numpy(denoised_image) * 255).astype(np.uint8))

    return np.mean(psnrs)


if __name__ == '__main__':
    ################################################################################
    # TODO: Your code goes here!
    ################################################################################

    # use the 'train' function with proper parameters for using biases and the
    # number of hidden layers
    model1 = train(sigma=0.1, use_bias=True, hidden_channels=32, epochs=2, batch_size=32, plot_every=200)
    model2 = train(sigma=0.1, use_bias=True, hidden_channels=64, epochs=2, batch_size=32, plot_every=200)
    model3 = train(sigma=0.1, use_bias=False, hidden_channels=32, epochs=2, batch_size=32, plot_every=200)
    model4 = train(sigma=0.1, use_bias=False, hidden_channels=64, epochs=2, batch_size=32, plot_every=200)
    # after training pass the model to the 'evaluate_model' function to run
    # on the validation dataset
    for noise in [0.05, 0.1, 0.2]:
        psnr1 = evaluate_model(model1, sigma=noise, output_filename=f'out_bias_32_{noise}.png')
        psnr2 = evaluate_model(model2, sigma=noise, output_filename=f'out_bias_64_{noise}.png')
        psnr3 = evaluate_model(model3, sigma=noise, output_filename=f'out_nobias_32_{noise}.png')
        psnr4 = evaluate_model(model4, sigma=noise, output_filename=f'out_nobias_64_{noise}.png')
        print(f'noise={noise}', psnr1, psnr2, psnr3, psnr4)
