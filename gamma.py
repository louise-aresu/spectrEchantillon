import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial, hyperu

from sim.pkprime_distrib import *

def pkprime_mass_function(mu, alpha, beta, k, factor=20):
    m = factor*k
    prob = np.zeros(m)
    prob[:-1] = 1
    for n in range(m-2, 0, -1):
        prob[n-1] = 1/((n+alpha-1)*(n+beta-1)) * (n*(2*n+alpha+beta-1+alpha*beta/mu)*prob[n]
                                                  - n*(n+1)*prob[n+1])

    return prob[:k+1]/np.sum(prob[:m])

def pkprime_mass_function_hyper(mu, alpha, beta, k):
    n = np.arange(k+1)
    Gn = gamma(n+alpha)*gamma(n+beta)/(gamma(alpha)*gamma(beta)*factorial(n))
    return Gn*(alpha*beta/mu)**alpha * hyperu(n+alpha, 1+alpha-beta, alpha*beta/mu)

def plot_figure(I0, Li, Lx):
    y = np.zeros((N, N, M), dtype=int)

    _, y = pkprime_distrib(I0, Li, Lx, (N, N), M)

    plt.figure()
    plt.title(r'$I_0 = $' f'{I0:.0e};\t'
              r'$L_I = $' f'{Li};\t'
              r'$Lx = $' f'{Lx}')

    y_real = np.arange(np.max(y) + 2)
    yhist = np.histogram(y, bins=y_real, density=True)[0]
    plt.scatter(y_real[:-1], yhist)
    for i in y_real[:-1]:
        plt.plot([i, i], [0, yhist[i]], '--k')

    p = pkprime_mass_function_hyper(I0, Li, Lx, np.max(y))
    plt.errorbar(np.arange(np.max(y) + 1), p, 5 / (N * np.sqrt(M)) * np.sqrt(p * (1 - p)),
                 marker='+', color='orange', linestyle=':', capsize=5)
    plt.semilogy()
    #plt.savefig(f'note/graphs/modele_pk/I0_{I0:.0e}_Li_{Li}_Lx_{Lx}.png')

## Paramètres du détecteur
N = 128
M = 1500
for I0 in [1e-3]:
    for Li in [20]:
        for Lx in [10]:
            nbsim = 100
            momt = np.ndarray((nbsim, 5))
            mome = np.ndarray((nbsim, 5))
            for i in range(nbsim):
                _, y = pkprime_distrib(I0, Li, Lx, (N, N), M)

                m1t = I0
                m2t = I0**2/(Li*Lx) * (1+Li)*(1+Lx)
                m3t = m2t*I0/(Li*Lx) * (2+Li)*(2+Lx)
                m4t = m3t*I0/(Li*Lx) * (3+Li)*(3+Lx)
                vart = m1t*(1-m1t) + m2t
                momt[i, :] = [vart, m1t, m2t, m3t, m4t]

                mome[i, :] = [np.var(y), np.mean(y), np.mean(y*(y-1)), np.mean(y*(y-1)*(y-2)), np.mean(y*(y-1)*(y-2)*(y-3))]

            print(np.mean(mome, axis=0))
            print(np.var(mome, axis=0))
            plt.figure()
            plt.errorbar(np.mean(momt, axis=0), np.mean(mome, axis=0), 2/(np.sqrt(nbsim)) * np.sqrt(np.var(mome, axis=0)),
                         capsize=5, linestyle='none', marker='x', color='orange')
            plt.axline((0,0), slope=1, color='k', linestyle='--')

plt.show()
# Corr = np.zeros((M, M))
# for t1 in range(M-1, -1, -1):
#     for t2 in range(M):
#         Corr[t1, t2] = np.mean(y[:,:,t1] * y[:,:,t2])
#
# plt.figure()
# im1 = plt.imshow(Corr, origin='lower')
# plt.colorbar(im1)
# plt.show()