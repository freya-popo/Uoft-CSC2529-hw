###################################################
#   Release code for CSC2529 HW6, task 1
#
#   Instructions:
#       Fill in the missing parts to compute the
#       anisotropic and isotropic TV terms below.
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/2022
###################################################

# import packages
import numpy as np
from pypher.pypher import psf2otf
from tqdm import tqdm
import torch


def deconv_adam_tv(b, c, lam, num_iters, learning_rate=5e-2, anisotropic_tv=True):
    # check if GPU is available, otherwise use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # otf of blur kernel and forward image formation model
    cFT = psf2otf(c, np.shape(b))
    cFT = torch.from_numpy(cFT).to(device)
    Afun = lambda x: torch.real(torch.fft.ifft2(torch.fft.fft2(x) * cFT))

    # finite differences kernels and corresponding otfs
    dx = np.array([[-1., 1.]])
    dy = np.array([[-1.], [1.]])
    dxFT = torch.from_numpy(psf2otf(dx, b.shape)).to(device)
    dyFT = torch.from_numpy(psf2otf(dy, b.shape)).to(device)
    dxyFT = torch.stack((dxFT, dyFT), axis=0)

    # convert b to PyTorch tensor
    b = torch.from_numpy(b).to(device)
    # initialize x and convert to PyTorch tensor
    x = torch.zeros_like(b, requires_grad=True).to(device)
    # initialize Adam optimizer
    optim = torch.optim.Adam(params=[x], lr=learning_rate)

    ################# begin task 1 ###################################

    # Define function handle to compute horizontal and vertical gradients.
    # You can use a local function definition using Python's lamda function
    # or write your own function for this. Use the convolutional image
    # formation in the Fourier domain to implement this using dxFT, dyFT,
    # or dxyFT, as discussed in the lecture and in the problem session.

    grad_fn = lambda x: torch.real(torch.fft.ifft2(torch.fft.fft2(x) * dxyFT))

    ################# end task 1 ###################################

    for it in tqdm(range(num_iters)):

        # set all gradients of the computational graph to 0
        optim.zero_grad()

        # this term computes the data fidelity term of the loss function
        loss_data = (Afun(x) - b).pow(2).sum()

        ################# begin task 1 ###################################

        # Complete these parts by calling the grad_fn function, which should
        # give you a full-resolution tensor with the gradients in x and y.
        # Then aggregate these gradients into a single scalar, i.e., the
        # TV pseudo-norm here and store the result in loss_regularizer

        # anisotropic TV term
        if anisotropic_tv:
            loss_regularizer = torch.norm(grad_fn(x), 1)  # you need to edit this, it's just a placeholder

        # isotropic TV term
        else:
            loss_regularizer = torch.norm(grad_fn(x), p=2, dim=0).sum()  # you need to edit this, it's just a placeholder

        ################# end task 1 ###################################

        # compute weighted sum of data fidelity and regularization term
        loss = loss_data + lam * loss_regularizer

        # compute backwards pass
        loss.backward()

        # take a step with the Adam optimizer
        optim.step()

    # return the result as a numpy array
    return x.detach().cpu().numpy()
