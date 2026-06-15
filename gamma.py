import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial

from sim.pkprime_distrib import *


def pkprime_mass_function(mu, alpha, beta, k):
    m = 50*k
    prob = np.zeros(m)
    prob[:-1] = 1
    for n in range(m-2, 0, -1):
        prob[n-1] = 1/((n+alpha-1)*(n+beta-1)) * (n*(2*n+alpha+beta-1+alpha*beta/mu)*prob[n]
                                                  - n*(n+1)*prob[n+1])

    return prob[:k+1]/np.sum(prob[:m])


## Paramètres du détecteur
N = 128
M = 1500

## Intensité du faisceau
I0 = 1e-2

## Nombres de vues
Li = 5
Lx = 2



x = np.zeros((N, N, M))
y = np.zeros((N, N, M), dtype=int)

for m in range(M):
    I = I0 * gamma_law(Li, 1)
    x[:, :, m], y[:, :, m] = pkprime_distrib(I, Li, Lx, (N, N))

#for m in range(M):
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # im1 = ax1.imshow(x[:,:,m], cmap='gray')
    # plt.colorbar(im1, ax=ax1, shrink=0.5, format='%.0e')
    # ax1.set_title('Intensity on sensor\n'
    #             r'<X>: ' f'{np.mean(x[:,:,m]):.2e}\n'
    #             r'V[X]: ' f'{np.var(x[:,:,m]):.2e}\n'
    #             r'sum: ' f'{np.sum(x[:,:,m]):.2f}\n')
    #
    # im2 = ax2.imshow(y[:, :, m], cmap='gray')
    # plt.colorbar(im2, ax=ax2, shrink=0.5, format='%.0e')
    # ax2.set_title('Photon count\n'
    #             r'<Y>: ' f'{np.mean(y[:,:,m]):.2e}\n'
    #             r'V[Y]: ' f'{np.var(y[:,:,m]):.2e}\n'
    #             r'sum: ' f'{np.sum(y[:,:,m]):.2f}\n')

fig, (ax1, ax2) = plt.subplots(1, 2)

x_real = np.arange(0, np.max(x)*101/100, np.max(x)/100)
ax1.hist(x.flatten(), density=True, stacked=True, bins=x_real, align='left')
ax1.set_title('Histogram of X')

y_real = np.arange(np.max(y)+2)
ax2.hist(y.flatten(), density=True, stacked=True, bins=y_real, align='left', rwidth=0.8)
ax2.set_title('Histogram of Y')

p = pkprime_mass_function(I0, Li, Lx, np.max(y))
ax2.errorbar(np.arange(np.max(y)+1), p, 3/(N*np.sqrt(M))*np.sqrt(p*(1-p)),
             marker='+', color='orange', linestyle=':', capsize=5)

ax2.semilogy()

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