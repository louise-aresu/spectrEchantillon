import numpy as np
import matplotlib.pyplot as plt

from sim.pg_distrib import *
from utils.math import *

## Paramètres du détecteur
N = 128
M = 100

# for I0 in [0.1, 1, 2]:
#     for Lx in [1, 4]:
#         x, y = sim_pg_distrib(I0, Lx, (N, N, M))
#
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,8))
#         fig.suptitle(f'I0: {I0}, Lx: {Lx}')
#         im1 = ax1.imshow(x[:,:,0], cmap='gray')
#         plt.colorbar(im1, ax=ax1, format='%.0e', shrink=0.8)
#         ax1.set_title('Intensity on sensor\n'
#                     r'<X>: ' f'{np.mean(x[:,:,0]):.2e}\n'
#                     r'V[X]: ' f'{np.var(x[:,:,0]):.2e}\n'
#                     r'sum: ' f'{np.sum(x[:,:,0]):.2f}\n')
#
#         im2 = ax2.imshow(y[:, :, 0], cmap='gray')
#         plt.colorbar(im2, ax=ax2, format='%.0e', shrink=0.8)
#         ax2.set_title('Photon count\n'
#                     r'<Y>: ' f'{np.mean(y[:,:,0]):.2e}\n'
#                     r'V[Y]: ' f'{np.var(y[:,:,0]):.2e}\n'
#                     r'sum: ' f'{np.sum(y[:,:,0]):.2f}\n')
#
#         x_real = np.arange(0, np.max(x), np.max(x)/100)
#         ax3.hist(x.flatten(), density=True, stacked=True, bins=x_real, align='left')
#         ax3.plot(x_real, gamma_mass_function(Lx, I0, x_real))
#         ax3.set_title('Histogram of intensity')
#
#         y_real = np.arange(0, np.ceil(np.max(y)), 1)
#         ax4.hist(y.flatten(), density=True, stacked=True, bins=y_real, align='left', rwidth=0.8)
#         ax4.set_title('Histogram of photon count')
#         ax4.plot(y_real, negbin_mass_function(Lx, I0, y_real))
#
#plt.show()

I0 = 0.05
Lx = 3

# nbsimu = 1
# res = 0.01 * np.ones((4, nbsimu))
#
# for order in range(2, 6):
#     for i in range(nbsimu):
#         _, y = sim_pg_distrib(I0, Lx, (N, N, M))
#         m1 = np.mean(y)
#         Y = 1
#         for o in range(order):
#             Y *= y-o
#         mo = np.mean(Y)
#
#         f = lambda alpha: 1 / (alpha ** order) * gamma(order + alpha) / gamma(alpha)
#         g = lambda alpha: f(alpha) - mo / (m1 ** order)
#         gprime = lambda alpha: f(alpha) * (-order / alpha + digamma(alpha + order) - digamma(alpha))
#
#         for _ in range(100):
#             res[order-2][i] -= g(res[order-2][i])/gprime(res[order-2][i])
#
# for order in range(2, 6):
#     print("Order ", order, ":")
#     print("Moyenne: ", np.mean(res[order-2]))
#     print("Variance: ", np.var(res[order-2]))
#
# res2 = np.zeros((nbsimu))
# for i in range(nbsimu):
#     _, y = sim_pg_distrib(I0, Lx, (N, N, M))
#     m1 = np.mean(y)
#     m2 = np.mean(y*(y-1))
#     res2[i] = m1**2/(m2-m1**2)
#
# print("Estimation par expression de L:")
# print("Moyenne: ", np.mean(res2))
# print("Variance: ", np.var(res2))

Lx = np.ones((M))
Lx[60:80] = 0.1

y = np.zeros((N, N, M))
for m in range(M):
    _, y[:,:,m] = sim_pg_distrib(I0, Lx[m], (N, N))

Corr = np.zeros((M, M))
for t1 in range(M-1, -1, -1):
    for t2 in range(M):
        if t1 == t2:
            Corr[t1, t2] = 1
        else:
            Corr[t1, t2] = np.mean(y[:,:,t1] * y[:,:,t2])/(np.mean(y[:,:,t1]) * np.mean(y[:,:,t2]))

plt.figure()
im1 = plt.imshow(Corr, origin='lower')
plt.colorbar(im1)
plt.show()