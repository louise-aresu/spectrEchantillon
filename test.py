import numpy as np
from sympy import *
from sympy.stats import *
from sympy import digamma
from utils.math import *
from sim.pg_distrib import *
import matplotlib.pyplot as plt

init_printing(use_unicode=True)

N = 128
M = 4500
I0 = 1e-2
Lx = 10

mu, L, k = symbols('mu L k', real=True)
K = NegativeBinomial("k", L, L/(mu+L))

def Esp(f, max=100):
    return Sum(f * density(K)(k), (k, 0, max))

InL = mu/L * 2 * polygamma(0, L) + mu**2*(L+1)/L**2 + mu/L + polygamma(0, L)**2 + Esp(polygamma(0, k+L)**2) -2*polygamma(0, L) * Esp(polygamma(0, k+L)) - 2/L * Esp(polygamma(0, k+L) * k)

print(1/(N**2*M*Esp((diff(ln(density(K)(k)), L))**2)).subs({mu:I0, L:Lx}).evalf())