import numpy as np
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import time as tm
from sim.pg_distrib import *
from utils.func import *

def moments_pg_distrib_estimation(y):
    # Calculation of first and second moments
    m1 = np.mean(y)
    m2 = np.mean(y**2)
    
    return (m1, m1**2 / (m2 - m1**2 - m1)) # (mu, L)

def monte_carlo_pg(I0, Lx, N, M, nbsimu, gen=None, log=True, hist=False, vals='variance'):
    mu = np.ndarray(nbsimu)
    L = np.ndarray(nbsimu)
    if log:
        times = np.ndarray(nbsimu)
        for n in range(nbsimu):
            start = tm.time()
            _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen=gen)
            mu[n], L[n] = moments_pg_distrib_estimation(y)
            end = tm.time()
            times[n] = end - start
            progress_bar(n+1, nbsimu, np.mean(times[:n+1]))
    else:
        for n in range(nbsimu):
            _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen=gen)
            mu[n], L[n] = moments_pg_distrib_estimation(y)

    I0e = np.mean(mu)
    Lxe = np.mean(L)

    if vals == 'variance':
        I0v = np.var(mu)
        Lxv = np.var(L)
    if vals == 'erreur':
        I0v = np.mean((mu - I0) ** 2)
        Lxv = np.mean((L - Lx) ** 2)


    if hist:
        plt.figure()
        plt.title(r'Estimations de $I_0$')
        plt.hist(mu, density=True, stacked=True, rwidth=0.9)
        plt.xlabel(r'$\hat{I_0}$')
        plt.figure()
        plt.title(r'Estimations de $L_X$')
        plt.hist(L, density=True, stacked=True, rwidth=0.9)
        plt.xlabel(r'$\hat{L_X}$')
    
    return ((I0e, I0v), (Lxe, Lxv))

if __name__ == "__main__":
    (_, I0v), (_, Lxv) = monte_carlo_pg(1e-2, 2, 128, 4500, 20)
    print(I0v, Lxv)