import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxpy as cp
from convex import generator
from data_generator import Generator
from consts import *
import torch
from utils import *
from cvxpylayers.torch import CvxpyLayer
from torch.nn.parameter import Parameter
torch.set_default_dtype(torch.double)




def f(x, w, b):
    return x @ w + b


def c(x, z, v):
    return cp.sum_squares(z-x)

Xdim = 57
z = cp.Variable(Xdim)
x = cp.Parameter(Xdim)
w = cp.Parameter(Xdim)
b = cp.Parameter(1)
v = np.random.randn(Xdim)


target = f(z, w, b) - c(x, z, v)
constraints = [z >= -10, z <= 10]
prob = cp.Problem(cp.Maximize(target), constraints)
layer = CvxpyLayer(prob, parameters=[x, w, b], variables=[z])

w_true = torch.randn(Xdim)
b_true = torch.randn(1)


X, Y = extract_data_from_file(r"C:\Users\sagil\Desktop\funcPred\spambase\spambase.data")
Xval, Yval = X[3500:4500], Y[3500:4500]
X, Y = X[:3500], Y[:3500]


print("percent of positive samples: {}%".format(100 * len(Y[Y > 0]) / len(Y)))

w_eval = torch.zeros_like(w_true, requires_grad=True)
b_eval = torch.zeros_like(b_true, requires_grad=True)


def loss(X, w, b, Y):
    # mse_loss = torch.nn.MSELoss()
    # return mse_loss(f(gen.layer(X, w, b)[0], w, b), Y)
    output = f(X, w, b)
    return torch.mean(torch.clamp(1 - output * Y, min=0))



val_losses, train_losses = fit(lambda X, Y: loss(X, w_eval, b_eval, Y), [w_eval, b_eval], X, Y, Xval, Yval,
                               opt=torch.optim.Adam, opt_kwargs={"lr": 1e-2},
                               batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)

w_eval.requires_grad = False
b_eval.requires_grad = False

plt.plot(np.arange(len(train_losses)), np.mean(train_losses, axis=1), color='b', label="train")
plt.plot(np.arange(len(val_losses)), val_losses, color='r', label="validation")

plt.show()

print("evaluated w, b:")
print(w_eval, b_eval)

# print("true w, b:")
# print(w_true, b_true)

Y_pred = torch.sign(Xval @ w_eval + b_eval)
evaluate_model(Yval, Y_pred)
