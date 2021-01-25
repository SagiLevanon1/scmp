import cvxpy as cp
import numpy as np
from consts import *

def f1(x, w):
    return np.dot(x, w)


def c1(x, z, v):
    return np.sum((x-z)**2 + v)

class generator:

    def __init__(self, f, c, dim):
        self.f = f
        self.c = c
        self.dim = dim

    def argmax_orig(self, x):
        candidates = np.random.rand(NUM_OF_CANDIDATES, self.dim)
        target = lambda z: self.f(z) - self.c(x, z)
        results = np.apply_along_axis(target, 1, candidates)
        index = np.argmax(results)
        return candidates[index]

    def argmax_cp(self, x, w, v):
        z = cp.Variable(self.dim)
        target = self.f(z, w) - self.c(x, z, v)
        constraints = [z >= 0, z <= 1]
        prob = cp.Problem(cp.Maximize(target), constraints)
        prob.solve()

        return z.value, prob.value

    def generate(self):
        z = cp.Variable(self.dim)
        x = cp.Parameter(self.dim)
        w = np.random.randn(self.dim)
        v = np.random.randn(self.dim)

        target = self.f(z, w) - self.c(x, z, v)
        constraints = [z >= 0, z <= 1]
        prob = cp.Problem(cp.Maximize(target), constraints)

        while 1:
            x.value = np.random.rand(self.dim)
            prob.solve()
            y = f1(x.value, w) - c1(x.value, x.value, v)
            yield x.value, y, z.value, prob.value