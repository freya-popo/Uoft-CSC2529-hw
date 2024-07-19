import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt


def grad_l2(A, x, b):
    # TODO: return the gradient of 0.5 * ||Ax - b||_2^2
    return A.T @ A @ x - A.T @ b

# def grad_l1(A, x, b):
#     K = A @ x - b
#     zero_entries = np.where(K ==
#     return 0.5 * A.transpose() @ div


def residual_l2(A, x, b):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def residual_l1(A, x, b):
    return 0.5 * np.linalg.norm(A @ x - b, ord=1)


def run_gd(A, b, step_size=1e-4, num_iters=1500, grad_fn=grad_l2, residual=residual_l1):
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
    progress_bar = tqdm(total=num_iters)
    timing = []
    residuals = []
    start_time = time()
    for iter in range(num_iters):
        resi = residual(A, x, b)
        x = x - step_size * grad_fn(A, x, b)
        progress_bar.update(1)
        residuals.append(resi)
        end_time = time()
        timing.append(end_time - start_time)
    return x, residuals, timing


def run_sgd(A, b, step_size=1e-4, num_iters=1500, batch_size = 10, grad_fn=grad_l2, residual=residual_l1):
    x = np.zeros((A.shape[1], 1))

    progress_bar = tqdm(total=num_iters)
    timing = []
    start_time = time()
    residuals = []
    for iter in range(num_iters):
        resi = residual(A, x, b)
        batch = np.random.randint(0, A.shape[0], batch_size)
        x = x - step_size * grad_fn(A[batch, :], x, b[batch])
        progress_bar.update(1)
        residuals.append(resi)
        end_time = time()
        timing.append(end_time - start_time)
    return x, residuals, timing

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
    print(resid_lsq)


    # TODO: implement and call your GD and SGD functions
    x, residual, timing = run_gd(A = A, b = b)
    # print(residual[-1])
    x_sgd_10, residual_sgd_10, timing_sgd_10 = run_sgd(A = A, b = b, num_iters=1500, batch_size=10)
    x_sgd_100, residual_sgd_100, timing_sgd_100 = run_sgd(A = A, b = b, num_iters=1500, batch_size=100)
    x_sgd_1000, residual_sgd_1000, timing_sgd_1000 = run_sgd(A = A, b = b, num_iters=1500, batch_size=1000)

    print(residual_sgd_1000[-1])
    plt.plot(timing, residual, label = 'gd')
    plt.plot(timing_sgd_10,residual_sgd_10, label = 'sgd(batch size = 10)')
    plt.plot(timing_sgd_100, residual_sgd_100, label = 'sgd(batch size = 100)')
    plt.plot(timing_sgd_1000, residual_sgd_1000, label = 'sgd(batch size = 1000)')
    plt.xlabel("time(s)")
    plt.ylabel("residual")
    plt.xscale('log')

    plt.legend()
    plt.show()
    # plt.plot(range(1, len(residual)+1), np.full((len(residual)), resid_lsq), label = 'svd')

    plt.plot(range(1, len(residual)+1), residual, label='gd')
    plt.plot(range(1, len(residual)+1), residual_sgd_10, label='sgd(batch size = 10)')
    plt.plot(range(1, len(residual)+1), residual_sgd_100, label='sgd(batch size = 100)')
    plt.plot(range(1, len(residual)+1), residual_sgd_1000, label='sgd(batch size = 1000)')
    plt.xlabel("step")
    plt.ylabel("residual")
    plt.yscale('log')

    plt.legend()
    plt.show()