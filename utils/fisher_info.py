import numpy as np
from sympy import *
from sympy.stats import *



def negbinomial_fisher_gen():
    mu, L, k = symbols('mu L k')

    p = L / (mu + L)
    K = NegativeBinomial("k", L, p)

    lB = diff(ln(beta(L, k)), L, L)

    return Matrix([[E(lB/k*K),-1/p],
                  [-1/p, L/(p**2*(1-p))]]), mu, L, k

def negbinomial_varmin(params, I0, Lx, N, M):
    IFisher, mu, L, k = params
    return ((IFisher ** -1) / (N ** 2 * M)).diagonal().evalf(subs={mu: I0, L: Lx})

