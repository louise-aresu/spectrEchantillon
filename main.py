import numpy as np
import matplotlib.pyplot as plt

from sim.pg_distrib import *
from estim.moments import *
from utils.math import *

## Paramètres du détecteur
N = 128
M = 1500

for I0 in [1e-2, 1e-1, 1e0, 1e1]:
    for Lx in [1e0, 1e1]:
        x, y = sim_pg_distrib(I0, Lx, (N, N, M))

        fig, ((topx, topy), (botx, boty)) = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})
        plt.subplots_adjust(wspace=0.5, hspace=None)

        plt.suptitle(r"$I_0 = $" f"{I0:.2e}\n"
                     r"$L_X = $" f"{Lx:.2e}")

        imgy = topy.imshow(y[:, :, 0], cmap='gray')
        plt.colorbar(imgy, ax=topy, format='%.0e', shrink=1)
        topy.set_title(r'$\widebar{Y} : $' f'{np.mean(y[:,:,:]):.2e}\n'
                       r'$\text{Var}(Y): $' f'{np.var(y[:,:,:]):.2e}')

        y_real = np.arange(0, np.ceil(np.max(y)), 1)
        boty.hist(y.flatten(), density=True, stacked=True, bins=y_real, align='left', rwidth=0.8)
        boty.plot(y_real, negbin_mass_function(Lx, I0, y_real))

        imgx = topx.imshow(x[:, :, 0], cmap='gray')
        plt.colorbar(imgx, ax=topx, format='%.0e', shrink=1)
        topx.set_title(r'$\widebar{X} : $' f'{np.mean(x[:, :, :]):.2e}\n'
                       r'$\text{Var}(X): $' f'{np.var(x[:, :, :]):.2e}')

        x_real = np.linspace(0, np.max(x), 100)
        botx.hist(x.flatten(), density=True, stacked=True, bins=x_real, align='left', rwidth=0.8)
        botx.plot(x_real, gamma_mass_function(Lx, I0, x_real))

plt.show()

# I0 = 10
# Lx = 1
#
# nbesti = 1
# estI0 = []
# estLx = []
#
# for i in range(nbesti):
#     _, y = sim_pg_distrib(I0,Lx, (N, N, M))
#     #estim = moments_pg_distrib_estimation(y)
#     m1 = np.mean(y)
#     m2 = np.mean(y**2)
#     estI0.append(m1)
#     estLx.append(m1**2 / (m2 - m1**2 - m1))
#
# I0bar = np.mean(estI0)
# Lxbar = np.mean(estLx)
# I0std = np.std(estI0, ddof=1)
# Lxstd = np.std(estLx, ddof=1)
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.set_title(r"Estimation de $I_0$", size="xx-large")
# ax1.hist(estI0, bins=np.linspace(I0*99/100, I0*101/100, 10), rwidth=0.9, align="mid")
# ax1.text(0.01, 0.99,
#          r"$\widebar{\hat{I_0}}$: " f"{I0bar:.2e}\n"
#          r"$\sigma$: " f"{I0std:.2e}\n"
#          f"Biais: {np.abs(I0bar-I0): .2e}",
#          ha='left', va='top', transform=ax1.transAxes, size="xx-large",
#          bbox=dict(facecolor='w', edgecolor='k'))
#
# ax2.set_title(r"Estimation de $L_x$", size="xx-large")
# ax2.hist(estLx, bins='auto', rwidth=0.9, align='mid')
# ax2.text(0.01 , 0.99,
#          r"$\widebar{\hat{L_X}}$: " f"{Lxbar:.2e}\n"
#          r"$\sigma$: " f"{Lxstd:.2e}\n"
#          f"Biais: {np.abs(Lxbar-Lx): .2e}",
#          ha='left', va='top', transform=ax2.transAxes, size="xx-large",
#          bbox=dict(facecolor='w', edgecolor='k'))
#
# fig.tight_layout()
# plt.show()