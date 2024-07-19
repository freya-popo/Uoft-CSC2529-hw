###################################################
#   Release code for CSC2529 HW6, task 3
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529  10/2022
###################################################

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm

# this function implements the finite differences method
from finite_differences import *


def admm_tv(b, Afun, Atfun, lam, rho, num_iters, imageResolution, anisotropic_tv=True):
    # initialize x,z,u with all zeros
    x = np.zeros(imageResolution)
    z = np.zeros((2, imageResolution[0], imageResolution[1]))
    u = np.zeros((2, imageResolution[0], imageResolution[1]))

    Ahat = lambda x: Atfun(Afun(x.reshape(imageResolution))) + rho * opDtx(opDx(x.reshape(imageResolution)))

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

        bhat = Atfun(b) + rho * opDtx(v)  # 64*64
        bhat = bhat.reshape(-1)  # 4096
        N = bhat.shape[0]
        A_operator = LinearOperator((N, N), matvec=Ahat)
        x = cg(A=A_operator, b=bhat, tol=cg_tolerance,
               maxiter=cg_iters)  # you need to edit this, it's just a placeholder
        x = x[0].reshape(imageResolution)

        ################# end task 3 ###################################

        # z update - soft shrinkage
        kappa = lam / rho
        v = opDx(x) + u

        # proximal operator of anisotropic TV term
        if anisotropic_tv:
            z = np.maximum(1 - kappa / np.abs(v), 0) * v

        # proximal operator of isotropic TV term
        else:
            vnorm = np.sqrt(v[0, :, :] ** 2 + v[1, :, :] ** 2)
            z[0, :, :] = np.maximum(1 - kappa / vnorm, 0) * v[0, :, :]
            z[1, :, :] = np.maximum(1 - kappa / vnorm, 0) * v[1, :, :]

        # u-update
        u = u + opDx(x) - z

    return x
