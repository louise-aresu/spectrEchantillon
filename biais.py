import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from estim.moments import *
from utils.fisher_info import *

I0 = 1e-2
Lx = 5
N = 128
nbsimu = 15

Ms = np.array([1500, 4500, 7500, 15000, 40000])

def sim(m):
    _, (Lxe, Lxv) = monte_carlo_pg(I0, Lx, N, m, nbsimu, vals='variance', log=False)
    return np.abs(Lxe - Lx), Lxv

if __name__ == "__main__":
    with mp.Pool() as pool:
        result = pool.map(sim, Ms)

    result = np.array(result).reshape((len(Ms), 2))
    biais = result[:, 0]
    erreur = result[:, 1]

    plt.figure()
    plt.title('Biais en fonction du nombre de frames')
    plt.plot(Ms, biais, 'b-', marker='+')
    plt.semilogx()
    plt.semilogy()
    plt.xlabel(r'Nombre de frames')
    plt.ylabel(r'Biais')

    params = negbinomial_fisher_gen()
    varmin = np.ndarray(len(Ms))
    for m in range(len(Ms)):
        varmin[m], _ = negbinomial_varmin(params, I0, Lx, N, Ms[m])

    plt.figure()
    plt.title('Variance en fontion du nombre de frames')
    plt.plot(Ms, erreur/varmin, 'r-', marker='+')
    plt.semilogx()
    plt.semilogy()
    plt.xlabel(r'Nombre de frames')
    plt.ylabel(r'Rapport variance de monte-carlo et CR')

    plt.show()