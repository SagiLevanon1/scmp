import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvxpy as cp
from convex import generator
from data_generator import Generator
from consts import *
import torch
from utils import fit

torch.set_default_dtype(torch.double)


def f(x, w):
    return x @ w


def c(x, z):
    return cp.sum_squares(z-x)


torch.manual_seed(0)

gen = Generator(f, c, XDIM)
w_true = torch.randn(XDIM)
X, Z = gen.get_data(200, w_true)
Xval, Zval = gen.get_data(100, w_true)

w_eval = torch.zeros_like(w_true)
w_eval.requires_grad_(True)

def loss(X, w, Z):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(gen.layer(X, w)[0], Z)


val_losses, train_losses = fit(lambda X, Z: loss(X, w_eval, Z), [w_eval], X, Z, Xval, Zval,
                               opt=torch.optim.Adam, opt_kwargs={"lr": 1e-1},
                               batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)

w_eval.requires_grad = False


plt.plot(np.arange(len(train_losses)), np.mean(train_losses, axis=1), color='b', label="train")
plt.plot(np.arange(len(val_losses)), val_losses, color='r', label="validation")

plt.show()

print("new w:")
print(w_eval)

print("true w:")
print(w_true)

Zval_eval = gen.layer(Xval, w_eval)[0]

X = Xval.numpy()
X_Prime_true = Zval.numpy()
X_Prime_eval = Zval_eval.numpy()

Y = f(Xval, w_true).numpy()
Y_Prime_true = f(Zval, w_true).numpy()
Y_Prime_eval = f(Zval_eval, w_true).numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y, marker='^', label='x')
ax.scatter(X_Prime_true[:, 0], X_Prime_true[:, 1], Y_Prime_true, marker='o', label='x\' true')
ax.scatter(X_Prime_eval[:, 0], X_Prime_eval[:, 1], Y_Prime_eval, marker='x', label='x\' estimated')

plt.show()

