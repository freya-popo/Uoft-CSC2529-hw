###################################################
#   Release code for CSC2529 HW6, task 3
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/2022
###################################################

import numpy as np

from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm

# PyTorch
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DnCNN model from Kai Zhang
from network_dncnn import DnCNN as net


def admm_dncnn(b, Afun, Atfun, lam, rho, num_iters, imageResolution):
    # initialize x,z,u with all zeros
    x = np.zeros(imageResolution)
    z = np.zeros(imageResolution)
    u = np.zeros(imageResolution)

    # load pre-trained DnCNN model
    model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
    model.load_state_dict(torch.load('dncnn_25.pth'), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    Ahat = lambda x: Atfun(Afun(x.reshape(imageResolution))) + rho * x.reshape(imageResolution)

    for it in tqdm(range(num_iters)):
        # x update using cg solver
        v = z - u

        cg_iters = 25  # number of iterations for CG solver
        cg_tolerance = 1e-12  # convergence tolerance of cg solver

        ################# begin task 3 ###################################
        #   Your task: implement a matrix-free conjugate gradient solver
        #       using the scipy.sparse.linalg.cg function in combination
        #       with the provided function handles Afun and Atfun to
        #       implement the x-update of ADMM. Use cg_iters as the number
        #       of iterations of cg and cg_tolerance as the "tol" parameters
        #       for the cg function.
        #
        #   Be careful with your image dimensions. The cg function expects
        #   vector inputs and outputs, whereas b, x, Afun, Atfun all
        #   work with 2D images. So make sure you work with the vectorized
        #   versions for the function handles you pass into cg and then reshape
        #   the result to a 2D image again after.

        bhat = Atfun(b) + rho * v  # 64*64

        bhat = bhat.reshape(-1)  # 4096
        N = bhat.shape[0]
        A_operator = LinearOperator((N, N), matvec=Ahat)
        x = cg(A=A_operator, b=bhat, tol=cg_tolerance,
               maxiter=cg_iters)  # you need to edit this, it's just a placeholder
        x = x[0].reshape(imageResolution)

        ################# end task 3 ###################################

        # z-update using DnCNN denoiser
        v = x + u
        v_tensor = torch.reshape(torch.from_numpy(v).float().to(device), (1, 1, v.shape[0], v.shape[1]))
        v_tensor_denoised = model(v_tensor)
        z = torch.squeeze(v_tensor_denoised).cpu().numpy()

        # u update
        u = u + x - z

    return x
