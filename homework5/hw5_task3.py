import imageio.v2
import skimage
import torch
from torch.fft import fft2, ifft2
import numpy as np
from hw5_task2 import BSDS300Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from models import Unet
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BlurredBSDS300Dataset(BSDS300Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True,
                 kernel_size=7, sigma=2, return_kernel=True):
        super(BlurredBSDS300Dataset, self).__init__(root, patch_size, split)

        # trim images to even size
        self.images = self.images[..., :-1, :-1]
        self.kernel_size = kernel_size
        self.return_kernel = return_kernel

        # extract blur kernel (use an MNIST digit)
        self.kernel_dataset = MNIST('./', train=True, download=True,
                                    transform=Compose([Lambda(lambda x: np.array(x)),
                                                       ToTensor(),
                                                       Lambda(lambda x: x / torch.sum(x))]))

        kernels = torch.cat([x[0] for (x, _) in zip(self.kernel_dataset, np.arange(self.images.shape[0]))])
        kernels = torch.nn.functional.interpolate(kernels[:, None, ...], size=2 * (kernel_size,))
        kernels = kernels / torch.sum(kernels, dim=(-1, -2), keepdim=True)
        self.kernel = kernels[[0]].repeat(kernels.shape[0], 1, 1, 1)

        # blur the images
        H = psf2otf(self.kernel, self.images.shape)
        self.blurred_images = ifft2(fft2(self.images) * H).real
        self.blurred_patches = self.patchify(self.blurred_images, patch_size)

        # save which blur kernel is used for each image
        self.patch_kernel = self.kernel.repeat(1, len(self.blurred_patches) // len(self.images), 1, 1)
        self.patch_kernel = self.patch_kernel.view(-1, *self.kernel.shape[-2:])

        # reshape kernel
        self.kernel = self.kernel.squeeze()

    def get_kernel(self, kernel_size, sigma):
        kernel = self.gaussian(kernel_size, sigma)
        kernel_2d = torch.matmul(kernel.unsqueeze(-1), kernel.unsqueeze(-1).t())
        return kernel_2d

    def __getitem__(self, idx):
        out = [self.blurred_images[idx][None, ...].to(device),
               self.images[idx][None, ...].to(device)]
        if self.return_kernel:
            out.append(self.kernel[[idx]].to(device))

        return out


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def psf2otf(psf, shape):
    inshape = psf.shape
    psf = torch.nn.functional.pad(psf, (0, shape[-1] - inshape[-1], 0, shape[-2] - inshape[-2], 0, 0))

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    psf = torch.roll(psf, shifts=(-int(inshape[-1] / 2), -int(inshape[-2] / 2)), dims=(-1, -2))

    # Compute the OTF
    otf = fft2(psf)

    return otf


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt) ** 2).mean().item())
    return out


def wiener_deconv(x, kernel):
    snr = 100  # use this SNR parameter for your results
    H = psf2otf(kernel, x.shape).to(device)
    G = torch.conj(H) * 1 / (1 / snr + H * torch.conj(H)).to(device)
    return ifft2(fft2(x) * G).real


def load_models():
    model_deblur_denoise = Unet().to(device)
    model_deblur_denoise.load_state_dict(torch.load('pretrained/deblur_denoise.pth', map_location=device))

    model_denoise = Unet().to(device)
    model_denoise.load_state_dict(torch.load('pretrained/denoise.pth', map_location=device))

    return model_deblur_denoise, model_denoise


def evaluate_model():
    # create the dataset
    dataset = BlurredBSDS300Dataset(split='test')

    # load the models
    model_deblur_denoise, model_denoise = load_models()

    # put into evaluation mode
    model_deblur_denoise.eval()
    model_denoise.eval()

    for sigma in [0.005, 0.01, 0.02]:
        psnr1 = 0
        psnr2 = 0
        psnr3 = 0
        i = 0
        for image, gt, kernel in dataset:
            img_noise = image + torch.randn_like(image) * sigma
            result_wiener = wiener_deconv(img_noise, kernel)
            result_net1 = model_deblur_denoise(img_noise)
            result_net2 = model_denoise(result_wiener)

            # save the psnrs
            psnr1 += calc_psnr(result_wiener, gt)
            psnr2 += calc_psnr(result_net1, gt)
            psnr3 += calc_psnr(result_net2, gt)
            i += 1
            # save out sample images to include in your writeup
            if i == 10 or i == 24:
                skimage.io.imsave(f'blur_image_{i}.png', (img_to_numpy(image) * 255).astype(np.uint8))
                skimage.io.imsave(f'noisy_image_{i}_sigma={sigma}.png',
                                  (img_to_numpy(img_noise) * 255).astype(np.uint8))
                skimage.io.imsave(f'image_{i}_method1_sigma={sigma}.png',
                                  (img_to_numpy(result_wiener) * 255).astype(np.uint8))
                skimage.io.imsave(f'image_{i}_method2_sigma={sigma}.png',
                                  (img_to_numpy(result_net1) * 255).astype(np.uint8))
                skimage.io.imsave(f'image_{i}_method3_sigma={sigma}.png',
                                  (img_to_numpy(result_net2) * 255).astype(np.uint8))
                print(i, f'when sigma = {sigma}', psnr1 / i, psnr2 / i, psnr3 / i)

        print(f'when sigma = {sigma}', psnr1 / i, psnr2 / i, psnr3 / i)
        # HINT: use the calc_psnr function to calculate the PSNR, and use the
        # wiener_deconv function to perform wiener deconvolution


if __name__ == '__main__':
    evaluate_model()
