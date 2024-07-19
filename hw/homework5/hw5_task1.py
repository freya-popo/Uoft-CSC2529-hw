import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import gradcheck
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.utils import make_grid
import unittest
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", "UserWarning")

"""
HW 5 Task 1 programming exercise

This exercise implements a small fully-connected neural network from scratch.
We implement custom layers with forward and backward passes for Linear and ReLU
layers, learn how to use autograd to evaluate gradients after a forward pass,
and use gradient descent to train the network to inpaint an image

Specifically you will need to complete the following tasks.

1. Fill in the code to calculate the derivatives for the backward pass for the
   LinearFunction and ReLUFunction.

2. Check your analytical gradient solutions from Task 1 part 3 of the homework
   against the gradients calcualated by AutoGrad. You will need to fill in the
   indicated portions of the "check_analytical" function. The gradients should
   match!

   How does autograd save computation in calculating these gradients
   compared to calculating the analytical gradients for W1 and b1 separately?

3. Use gradient descent to train your network to overfit an image inpainting
   task and learn to inpaint the missing values of an image. Complete and run the
   "train_network" function to do this.

"""


class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        # we will save the input, weight, and bias to help us calculate the
        # gradients in the backward pass
        ctx.save_for_backward(input, weight, bias)

        # return the output of the linear layer
        return input.mm(weight.T) + bias[None, :]

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve the saved variables from the context
        input, weight, bias = ctx.saved_tensors

        #######################################################################
        # TODO: complete these lines, replacing "None" with the correct
        # calculations
        #######################################################################

        # We need to return the gradients with respect to the input, the weight
        # matrix, and the bias vector.
        #
        # This can be done in two steps. (1) We need to compute the gradient of
        # the layer output with respect to the input, the weight matrix, and the
        # bias vector. These correspond to partial derivative terms in the chain
        # rule that you already derived in your homework. (2) We need to
        # multiply this by the "upstream" gradient (grad_output), which is the
        # result of multiplying all the previous terms in the chain rule that
        # have already been calculated in the backward pass (starting at the
        # loss function and flowing backwards).
        #
        # The backward function then returns each of these gradient values. The
        # gradients with respect to the weight and bias parameters will be used
        # for updating the parameters during training, and the value returned
        # for grad_input will become the new grad_output for the next term in
        # the chain rule as we continue the backward pass.

        grad_input = grad_output @ weight  # 1*n
        grad_weight = grad_output.T @ input  # n*n
        grad_bias = grad_output  # 1*n
        #######################################################################

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features):
        super(Linear, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        self.bias = nn.Parameter(torch.empty(output_features))

        # initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias)


class ReLUFunction(Function):

    @staticmethod
    def forward(ctx, input):
        # we will save the input, weight, and bias to help us calculate the
        # gradients in the backward pass
        ctx.save_for_backward(input)

        # return the output of the linear layer
        return torch.clamp(input, 0)

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve the saved variables from the context
        input, = ctx.saved_tensors

        #######################################################################
        # TODO: complete these lines, replacing "None" with the correct
        # calculations
        #######################################################################

        # The gradient with respect to the input will become the new
        # "grad_output" in the next backward operation as we continue the
        # backward pass. Also see comments above in the LinearFunction part of
        # the homework.
        # grad_output ... 1*n
        # matrix = np.zeros((len(grad_output), len(grad_output)))
        # print(grad_output.shape)
        # for i in range(len(grad_output)):
        #     for j in range(len(grad_output)):
        #         if i == j and input[i] >= 0:
        #             matrix[i][j] = 1
        #
        # grad_input = grad_output @ matrix  # 1*n
        # print(grad_output.shape, matrix.shape, grad_output.shape)
        # error: there are some batches here, we need to consider the dimensions of batches in each test
        # gradient of relu == gradient_output * relu'
        # relu' means when the input value>0 ==> relu'=1, else relu'=0
        # therefore the dimension of grad_input should be the same with that of grad_output
        # print(grad_output.shape, input.shape)
        grad_input = grad_output.clone()  #
        grad_input[input < 0] = 0
        # print(grad_input.shape)
        #######################################################################

        return grad_input


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return ReLUFunction.apply(input)


class FullyConnectedNet(nn.Module):
    def __init__(self, din=4, dout=3, n=8):
        super(FullyConnectedNet, self).__init__()

        self.din = din
        self.dout = dout
        self.n = n

        self.L1 = Linear(self.din, self.n)
        self.N1 = ReLU()
        self.L2 = Linear(self.n, self.n)
        self.N2 = ReLU()
        self.L3 = Linear(self.n, self.dout)

    def forward(self, x):
        self.g1 = self.L1(x)
        self.h1 = self.N1(self.g1)
        self.g2 = self.L2(self.h1)
        self.h2 = self.N2(self.g2)
        self.yhat = self.L3(self.h2)

        return self.yhat


def check_analytical_gradients():
    """
    Now that you've implemented autograd in PyTorch, let's see how we can use
    this to easily calculate gradients for all the network weights

    """

    # create the fully connected neural network
    net = FullyConnectedNet()

    # use double precision for our numerical checks
    net.double()

    # declare the input and "ground truth" output (arbitrary) we will compare to
    x = torch.randn(1, net.din, dtype=torch.double)
    y = torch.rand(1, net.dout, dtype=torch.double)

    # run the forward pass through the network
    yhat = net(x)

    # we'll use MSE loss
    loss = 1 / 2 * torch.sum((y - yhat) ** 2)

    # running .backward() will calculate the gradient of all the weights and
    # biases with respect to the loss. These are stored in the .grad property of
    # each parameter
    loss.backward()

    # Grab all the values we'll need to calculate the gradients analytically
    W3 = net.L3.weight
    g2 = net.g2
    W2 = net.L2.weight
    g1 = net.g1
    W1 = net.L1.weight
    b1 = net.L1.bias
    print(W1.shape)
    ###########################################################################
    # TODO: write your analytical expression for dloss/dW1 from the gradient descent
    # update in part 3
    # HINT: your expression will depend on the variables W3, g2, W2, g1, and x
    g2_g = g2.clone()
    g2_g[g2 >= 0] = 1
    g2_g[g2 < 0] = 0
    g1_g = g1.clone()
    g1_g[g1 >= 0] = 1
    g1_g[g1 < 0] = 0

    W1_grad = ((yhat - y) @ W3 * g2_g @ W2 * g1_g).T @ x  # replace this line

    # TODO: write your analytical expression for dloss/db1 from the gradient descent
    # update in part 3
    # HINT: your expression will depend on the variables W3, g2, W2, g1, and x

    b1_grad = (yhat - y) @ W3 * g2_g @ W2 * g1_g  # replace this line

    ###########################################################################

    # check to make sure that it matches autograd
    W1_autograd = W1.grad
    b1_autograd = b1.grad

    assert torch.allclose(W1_grad, W1_autograd), \
        'FAILED: Incorrect W1 analytical gradient!'
    print('PASSED: W1 Analytical Gradient')

    assert torch.allclose(b1_grad, b1_autograd), \
        'FAILED: Incorrect b1 analytical gradient!'
    print('PASSED: b1 Analytical Gradient')


def train_network(lr=2):
    # set up inpainting dataset, normalize by mean and std
    mnist_mean = 0.1307
    mnist_std = 0.3081
    dataset = MNIST('./', train=True, download=True,
                    transform=Compose([Lambda(lambda x: np.array(x)),
                                       ToTensor(),
                                       Normalize((mnist_mean,), (mnist_std,)),
                                       Lambda(lambda x: torch.flatten(x))
                                       ]))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # instantiate neural network
    net = FullyConnectedNet(784, 784, n=32)

    # overfit the network to inpaint this image using the ground truth
    losses = []
    iteration = 0
    N_epochs = 3

    def mask_pixels(x):
        x = x.reshape(-1, 28, 28)
        x[:, 14:, :] = 0.
        return x.reshape(-1, 28 ** 2)

    pbar = tqdm(total=N_epochs * len(dataloader))
    for epoch in range(N_epochs):
        for idx, (y, _) in enumerate(dataloader):

            # mask out pixels, we'll need to inpaint them
            x = y.clone()
            x = mask_pixels(x)

            # run a forward pass through the network
            yhat = net(x)

            # calculate the loss
            loss = torch.mean((yhat - y) ** 2)
            losses.append(loss.item())

            # run backward pass to calculate gradients for all the parameters
            loss.backward()

            ########################################################################
            # TODO: write the gradient descent update rule
            # HINT: update p.data using the learning rate and the
            # gradient stored in p.grad
            with torch.no_grad():
                for p in net.parameters():
                    # raise NotImplementedError('Need to write gradient descent rule')  # remove this line
                    p.data = p.data - lr * p.grad  # write gradient descent update rule here

            ########################################################################

            # set the gradients to zero (otherwise they will be accumulated in the
            # .grad array of each parameter during the iterations)
            net.zero_grad()

            if iteration % 1000 == 0:
                plt.ion()
                plt.clf()

                # plot ground truth
                plt.subplot(141)
                y_plot = (y[0].reshape(28, 28).detach().cpu().numpy() * mnist_std) + mnist_mean
                plt.imshow(y_plot, aspect='equal', cmap='gray')
                plt.clim((0, 1))
                plt.axis('off')
                plt.title('Ground Truth')

                # plot measurements
                plt.subplot(142)
                x_plot = (x[0].reshape(28, 28).detach().cpu().numpy() * mnist_std) + mnist_mean
                plt.imshow(x_plot, aspect='equal', cmap='gray')
                plt.clim((0, 1))
                plt.axis('off')
                plt.title('Measurements')

                # plot reconstructed image
                plt.subplot(143)
                yhat_plot = (yhat[0].reshape(28, 28).detach().cpu().numpy() * mnist_std) + mnist_mean
                plt.imshow(yhat_plot, aspect='equal', cmap='gray')
                plt.clim((0, 1))
                plt.axis('off')
                plt.title('Reconstructed Image')

                # plot loss
                plt.subplot(144)
                plt.plot(losses)
                plt.yscale('log')
                plt.title('Loss')
                plt.xlabel('iterations')
                plt.ylim((2e-1, 2))
                plt.gca().set_aspect(1. / plt.gca().get_data_ratio())

                plt.tight_layout()
                plt.pause(0.1)

            iteration += 1
            pbar.update(1)

    # print some examples
    N_examples = 10
    list_y = []
    list_x = []
    list_yhat = []
    dataloader = DataLoader(dataset, batch_size=N_examples, shuffle=True)

    y, _ = iter(dataloader).next()

    # mask out pixels, we'll need to inpaint them
    x = y.clone()
    x = mask_pixels(x)

    # run a forward pass through the network
    list_y.append(y.reshape(-1, 1, 28, 28))
    list_yhat.append(net(x).reshape(-1, 1, 28, 28))
    list_x.append(x.reshape(-1, 1, 28, 28))

    # plot reconstructed image
    images = torch.cat([*list_y, *list_x, *list_yhat], dim=0) * mnist_std + mnist_mean
    grid = make_grid(images, nrow=N_examples, padding=2, pad_value=1)
    grid = (grid.detach().cpu().numpy()[0])
    plt.imshow(grid, aspect='equal', cmap='gray')
    plt.clim((0, 1))
    plt.axis('off')
    plt.title('Reconstructed Images')


class HW5Checker(unittest.TestCase):

    def test_linear(self):
        input = (torch.randn(64, 16, dtype=torch.double, requires_grad=True),
                 torch.randn(32, 16, dtype=torch.double, requires_grad=True),
                 torch.randn(32, dtype=torch.double, requires_grad=True))

        result = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4,
                           raise_exception=False)
        assert result, "Incorrect Linear backward pass"
        print('PASSED: Linear Backward')

    def test_relu(self):
        input = (torch.randn(64, 16, dtype=torch.double, requires_grad=True))

        result = gradcheck(ReLUFunction.apply, input, eps=1e-6, atol=1e-4,
                           raise_exception=False)
        assert result, "Incorrect ReLU backward pass"
        print('PASSED: ReLU Backward')

    def test_net(self):
        net = FullyConnectedNet()
        net.double()
        input = torch.randn(16, net.din, dtype=torch.double).requires_grad_()

        result = gradcheck(net, (input,), eps=1e-6, atol=1e-4, raise_exception=False)
        assert result, "Incorrect FullyConnectedNet backward pass"
        print('PASSED: FullyConnectedNet Backward')

    def test_analytical_grad(self):
        check_analytical_gradients()


def check_part_a():
    suite = unittest.TestSuite()
    suite.addTest(HW5Checker('test_linear'))
    suite.addTest(HW5Checker('test_relu'))
    suite.addTest(HW5Checker('test_net'))

    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if len(result.failures) == 0:
        print('ALL TESTS PASSED')
        return True
    else:
        return False


def check_part_b():
    suite = unittest.TestSuite()
    suite.addTest(HW5Checker('test_analytical_grad'))

    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if len(result.failures) == 0:
        print('ALL TESTS PASSED')
        return True
    else:
        return False


if __name__ == '__main__':
    check_part_a()
    check_part_b()
    train_network()
