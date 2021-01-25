import cvxpy as cp
import numpy as np
from consts import *
import torch
from cvxpylayers.torch import CvxpyLayer
import cvxpylayers
#import torch

torch.set_default_dtype(torch.double)


class Generator:

    def __init__(self, f, c, Xdim):
        self.f = f
        self.c = c
        self.Xdim = Xdim

        z = cp.Variable(self.Xdim)
        x = cp.Parameter(self.Xdim)
        w = cp.Parameter(self.Xdim)
        # v = cp.Parameter(self.Xdim)

        # target = self.f(z, w) - self.c(x, z, v)
        target = self.f(z, w) - self.c(x, z)
        constraints = [z >= -10, z <= 10]
        prob = cp.Problem(cp.Maximize(target), constraints)
        # self.layer = CvxpyLayer(prob, parameters=[x, w, v], variables=[z])
        self.layer = CvxpyLayer(prob, parameters=[x, w], variables=[z])


    # def get_data(self, N, w_true, v_true):
    #     X = torch.randn(N, self.Xdim)
    #     Y = self.layer(X, w_true, v_true)[0]
    #     return X, Y

    def get_data(self, N, w_true):
        X = torch.randn(N, self.Xdim)
        Y = self.layer(X, w_true)[0]
        return X, Y


# def f(x, w):
#     return w+x
# n = 20
# m = 10
#
# z = cp.Variable(2)
# x = cp.Parameter(2)
# w = cp.Parameter(2)
#
# objective = cp.sum(z@f(x, w))
# constraints = [z <= 1, z >= 0]
# prob = cp.Problem(cp.Minimize(objective), constraints)
# layer = CvxpyLayer(prob, [x, w], [z])
#
#
# def get_data(N, w):
#     X = torch.randn(N, 2)
#     Y = layer(X, w)[0]
#     return X, Y
#
# torch.manual_seed(0)
# theta_true = torch.randn(2)
# X, Y = get_data(100, theta_true)
#
# print(X.size())
# print(Y.size())

