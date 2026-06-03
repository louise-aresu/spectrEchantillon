import numpy as np
import matplotlib.pyplot as plt

from utils.math import *

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
for ax, mu in zip([ax1, ax2, ax3, ax4], [1e-2, 1e-1, 1e0, 1e1]):
    k = np.arange(0, 20)
    ax.set_ylim(0, 1)
    ax.scatter(k, poisson_mass_function(mu, k))
    for x in k:
        ax.plot((x, x), (0, poisson_mass_function(mu, x)), '--k')
    ax.text(7, 0.6, r'$\mathbf{\lambda =}$' f'{mu : .2f}', fontsize=20, weight='bold')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
mu = 1
for ax, L in zip([ax1, ax2, ax3, ax4], [1, 2, 5, 10]):
    x = np.linspace(0, 5, 100)
    ax.plot(x, gamma_mass_function(L, mu, x))
    ax.set_ylim(0, 1.4)
    ax.text(3, 1, r'$\mathbf{L =}$' f'{L}', fontsize=20, weight='bold')
plt.show()