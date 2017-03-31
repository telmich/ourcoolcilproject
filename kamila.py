import numpy as np

def grad_i(d, n, X, U, Z):
    """ docstring here to describe what I do ;-)"""

    # I want: partial(X_dn, d, n, U, Zt)
    return TODO

def stochastic_gradient(Omega, X, k, t, nu):
    """According to http://www.da.inf.ethz.ch/teaching/2017/CIL/material/lecture/CIL2017-03-Optimization.pdf slide 19.

    k: the smaller dimension of the U and Z that we want / the (maximum) rank of U and Z
    """

    # 1. initialize with something smart
    # TODO something smart ;-)
    d, n = X.shape
    U, Z = np.random.rand(d, k), np.random.rand(n, k)
    for t = range(steps):
        TODO
