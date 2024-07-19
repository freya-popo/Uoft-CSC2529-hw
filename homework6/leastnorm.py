###################################################
#   Release code for CSC2529 HW6, task 3
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/2022
###################################################

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from tqdm import tqdm


def leastnorm(b, Afun, Atfun, num_iters, imageResolution):
    # number of measurements
    N = b.shape[0]
    # convergence tolerance of cg solver
    cg_tolerance = 1e-12

    # initialize x with all zeros
    x = np.zeros(imageResolution)

    ################# begin task 3 ###################################
    #   Your task: implement a matrix-free conjugate gradient solver
    #       using the scipy.sparse.linalg.cg function in combination
    #       with the provided function handles Afun and Atfun. Use
    #       num_iters as the number of iterations of cg and
    #       cg_tolerance as the "tol" parameters for the cg function.
    #
    #   Hints:
    #       1. solve the problem (AA')y = b using CG first (CG only
    #           works for positive semi-definite matrices, like AA')
    #       2. then multiply the result by A' to get x as x=A'y
    #
    #   Be careful with your image dimensions. The cg function expects
    #   vector inputs and outputs, whereas b, x, Afun, Atfun all
    #   work with 2D images. So make sure you work with the vectorized
    #   versions for the function handles you pass into cg and then reshape
    #   the result to a 2D image again after.
    method = lambda x: Afun(Atfun(x))
    A_operator = LinearOperator((N, N), matvec=method)
    u = cg(A=A_operator, b=b, tol=cg_tolerance, maxiter=num_iters)
    x = Atfun(u[0])  # you need to edit this, it's just a placeholder

    ################# end task 3 ###################################

    return x
