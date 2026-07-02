import numpy as np
from sympy import *
from sympy.stats import *
from utils.math import *
from sim.pg_distrib import *
import matplotlib.pyplot as plt

init_printing(use_unicode=True)

N = 128
M = 1500
I0 = 5e-3
Lx = 4

mu, L, k = symbols('mu L k')

m1 = mu
m2 = mu ** 2 * (1 + 1 / L) + mu

m1m = diff(m1, mu)
m2m = diff(m2, mu)
m1L = diff(m1, L)
m2L = diff(m2, L)

J = Matrix([[m1m, m1L], [m2m, m2L]])

p = L / (mu + L)
K = NegativeBinomial("k", L, p)

print(simplify(variance(K)))

lB = diff(ln(beta(L, k)), L, L)

IFisher = Matrix([[E(lB/k*K),-1/p],
                  [-1/p, L/(p**2*(1-p))]])

print((IFisher**-1).evalf(subs={mu:1e-2, L:5}))
print(((J.transpose()*J)**-1).evalf(subs={mu:1e-2, L:5}))