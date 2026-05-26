import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial

from sim.pkprime_distrib import *

## Paramètres du détecteur
N = 1024
M = 5

## Intensité du faisceau
I0 = 20*N**2

## Nombres de vues
Lx = 1
Li = 3

x = np.zeros((N, N, M))
y = np.zeros((N, N, M))

for m in range(M):
    x[:, :, m], y[:, :, m] = pkprime_distrib(I0, Li, Lx, (N, N))

# for m in range(M):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#
#     im1 = ax1.imshow(x[:,:,m], cmap='gray')
#     plt.colorbar(im1, ax=ax1, shrink=0.5, format='%.0e')
#     ax1.set_title('Intensity on sensor\n'
#                 r'<X>: ' f'{np.mean(x[:,:,m]):.2e}\n'
#                 r'V[X]: ' f'{np.var(x[:,:,m]):.2e}\n'
#                 r'sum: ' f'{np.sum(x[:,:,m]):.2f}\n')
#
#     im2 = ax2.imshow(y[:, :, m], cmap='gray')
#     plt.colorbar(im2, ax=ax2, shrink=0.5, format='%.0e')
#     ax2.set_title('Photon count\n'
#                 r'<Y>: ' f'{np.mean(y[:,:,m]):.2e}\n'
#                 r'V[Y]: ' f'{np.var(y[:,:,m]):.2e}\n'
#                 r'sum: ' f'{np.sum(y[:,:,m]):.2f}\n')

mu = I0/N**2

fig, (ax1, ax2) = plt.subplots(1, 2)

x_real = np.arange(0, np.max(x), np.max(x)/100)
ax1.hist(x.flatten(), density=True, stacked=True, bins=x_real, align='left')
ax1.set_title('Histogram of X')

y_real = np.arange(0, np.ceil(np.max(y)), 1)
ax2.hist(y.flatten(), density=True, stacked=True, bins=y_real, align='left', rwidth=0.8)
ax2.set_title('Histogram of Y')
plt.show()