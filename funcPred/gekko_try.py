from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# XDIM = 2
# # GEKKO model
# m = GEKKO()
#
# x = m.Var(value=np.zeros(XDIM))
# w = m.Param(value=np.random.randn(XDIM))
# print(w[0], w[1])
#
# m.Equation(x <= 2)
# m.Equation(x >= -2)
# fx = w[0] * x[0] + w[1] * x[1]
# m.Obj(1/(1 + m.exp(-fx)))
# m.options.IMODE = 2
#
# m.solve(disp=False)
#
# print('Optimized, x = ' + str(x.value))

w = np.random.randn(2)
print(w)
res = optimize.fmin_l_bfgs_b(lambda x: -1/(1 + np.exp(-np.matmul(w, x))), np.zeros(2), approx_grad=True)
print(res)