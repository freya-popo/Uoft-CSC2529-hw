import numpy as np

from time import time
import matplotlib.pyplot as plt


def grad_l2(A, x, b):
    # TODO: return the gradient of 0.5 * ||Ax - b||_2^2
    return A.T @ A @ x - A.T @ b


def residual_l2(A, x, b):
    return 0.5 * np.linalg.norm(A @ x - b) ** 2


def run_gd(A, b, mode, step_size=1e-4, num_iters=1500, grad_fn=grad_l2, residual=residual_l2):
    ''' Run gradient descent to solve Ax = b

    Parameters
    ----------
    A : matrix of size (N_measurements, N_dim)
    b : observations of (N_measurements, 1)
    step_size : gradient descent step size
    num_iters : number of iterations of gradient descent
    grad_fn : function to compute the gradient
    residual : function to compute the residual

    Returns
    -------
    x
        output matrix of size (N_dim)

    residual
        list of calculated residuals at each iteration

    timing
        time to execute each iteration (should be cumulative to each iteration)

    '''

    # initialize x to zero
    x = np.zeros((A.shape[1], 1))

    # TODO: complete the gradient descent algorithm here
    # you can also complete and use the grad_l2 and residual_l2 functions

    # this function can return a list of residuals and timings at each iteration so
    # you can plot them and include them in your report

    # don't forget to also implement the stochastic gradient descent version of this function!

    ### gradiant decent
    if mode == 'GD':
        i = 0
        residuals = [residual(A, x, b)]
        times = []
        time_start = time()
        while i < num_iters:
            x -= step_size * grad_fn(A, x, b)
            residuals.append(residual(A, x, b))
            times.append((time() - time_start))
            i += 1
        return x, residuals, times
    elif mode == 'SGD':
        i = 0
        residuals = [residual(A, x, b)]
        times = []
        time_start = time()
        while i < num_iters:

            random_integers = np.random.randint(0, N_measurements, size=B)
            A1 = A[random_integers, :]
            b1 = b[random_integers]

            x -= step_size * grad_fn(A1, x, b1)

            residuals.append(residual(A, x, b))
            times.append((time() - time_start))
            i += 1
        return x, residuals, times


def run_lsq(A, b):
    ''' Numpy's implementation of least squares which uses SVD & matrix factorization from LAPACK '''
    x, resid, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x, 0.5 * resid


if __name__ == '__main__':
    # set up problem
    N_dim = 128  # size of x
    N_measurements = 16384  # number of measurements or rows of A

    # load matrix
    dat = np.load('task3.npy', allow_pickle=True)[()]

    # data matrix -- here the rows are measurements and columns are dimension N_dim
    A = dat['A']

    # corrupted measurements
    b = dat['b']

    # least squares solve using SVD + matrix factorization
    # you can compare your solution to this
    x_lsq, resid_lsq = run_lsq(A, b)
    iteration = 1500

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # TODO: implement and call your GD and SGD functions
    result1 = run_gd(A, b, 'GD', step_size=1e-4, num_iters=iteration, grad_fn=grad_l2, residual=residual_l2)
    ax[0].plot(range(iteration + 1), result1[1])
    ax[1].plot(result1[2], result1[1][1:])
    for B in [10, 100, 1000]:
        result2 = run_gd(A, b, 'SGD', step_size=1e-4, num_iters=iteration, grad_fn=grad_l2, residual=residual_l2)
        ax[0].plot(range(iteration + 1), result2[1])
        ax[1].plot(result2[2], result2[1][1:])
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[0].plot(range(iteration + 1), [resid_lsq] * (iteration + 1))
    ax[0].legend(['GD', 'SGD B=10', 'SGD B=100', 'SGD B=1000', 'SVD'])
    ax[1].legend(['GD', 'SGD B=10', 'SGD B=100', 'SGD B=1000'])
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('residual')
    ax[1].set_xlabel('time(s)')
    ax[1].set_ylabel('residual')
    plt.savefig('task3.png', bbox_inches='tight')
    plt.show()
