import numpy as np
import matplotlib.pyplot as plt

from sim.pg_distrib import *
from estim.moments import *
from utils.math import *

## Paramètres du détecteur
N = 128
M = 1500

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

I0 = 0.1
Lx = 0.5

nbesti = 100
estI0 = []
estLx = []

for i in range(nbesti):
    _, y = sim_pg_distrib(I0,Lx, (N, N, M))
    estim = moments_pg_distrib_estimation(y)
    estI0.append(estim[0])
    estLx.append(estim[1])

I0bar = np.mean(estI0)
Lxbar = np.mean(estLx)
I0var = np.var(estI0)
Lxvar = np.var(estLx)


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title(r"Estimation de $I_0$")
ax1.hist(estI0, bins='auto', rwidth=0.9, align='mid')
ax1.text(0.95, 0.95,
         r"$\widebar{\hat{I_0}}$: " f"{I0bar:.2e}\n"
         r"Var($\hat{I_0}$): " f"{I0var:.2e}\n"
         f"Biais: {np.abs(I0bar-I0): .2e}",
         ha='right', va='top', transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='k'))

ax2.set_title(r"Estimation de $L_x$")
ax2.hist(estLx, bins='auto', rwidth=0.9, align='mid')
ax2.text(0.95, 0.95,
         r"$\widebar{\hat{L_X}}$: " f"{Lxbar:.2e}\n"
         r"Var($\hat{L_X}$): " f"{Lxvar:.2e}\n"
         f"Biais: {np.abs(Lxbar-Lx): .2e}",
         ha='right', va='top', transform=ax2.transAxes, bbox=dict(facecolor='w', edgecolor='k'))

#fig.tight_layout()
plt.show()