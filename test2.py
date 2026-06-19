import numpy as np
from sympy import *
from sim.pg_distrib import *
import matplotlib.pyplot as plt
import time
import os

init_printing(use_unicode=True)

mu, L = symbols('mu L')

m1 = mu
m2 = mu ** 2 * (1 + 1 / L) + mu

m1m = diff(m1, mu)
m2m = diff(m2, mu)
m1L = diff(m1, L)
m2L = diff(m2, L)

J = Matrix([[m1m, m1L], [m2m, m2L]])
cov = (J.T * J) ** -1


# var = 1/((N**2)*M) * cov.diagonal().applyfunc(sqrt)
# std = var.applyfunc(sqrt)
# print(std.evalf(subs={mu:I0, L:Lx}))

def gen_vars(I0, Lx, N, M, theta0={mu: 0.01, L: 1}, nbsim=10):
    vals = np.ndarray((nbsim, 2))

    for i in range(nbsim):
        _, y = sim_pg_distrib(I0, Lx, (N, N, M))
        m1e = np.mean(y)
        m2e = np.mean(y ** 2)

        r = Matrix([[m1 - m1e], [m2 - m2e]])

        theta = theta0
        for _ in range(3):
            p = J.evalf(subs=theta).LUsolve(-r.evalf(subs=theta))
            # p = -J.evalf(subs=theta)**-1 * r.evalf(subs=theta)
            theta = {mu: theta[mu] + p[0], L: theta[L] + p[1]}
        vals[i] = [theta[mu], theta[L]]

    return np.var(vals[:, 0]), np.var(vals[:, 1])


I0s = np.linspace(0.001, 1e-2, 10)
Lxs = np.linspace(1.5, 4, 10)

Ivars = np.zeros((len(Lxs), len(I0s)))
Lvars = np.zeros((len(Lxs), len(I0s)))

counter = 0
maxcounter = len(Lxs) * len(I0s)
times = np.ndarray(maxcounter)

for iI0 in range(len(I0s)):
    for iLx in range(len(Lxs)):
        start = time.time()
        Ivars[iLx, iI0], Lvars[iLx, iI0] = gen_vars(5e-3, 4, 128, 1500, theta0={mu: I0s[iI0], L: Lxs[iLx]}, nbsim=10)
        end = time.time()
        times[counter] = end - start
        counter += 1
        mean_time = time.gmtime((maxcounter - counter) * np.mean(times[:counter]))
        progress_bar(counter, maxcounter, mean_time)

plt.figure()
plt.title(r'$I_0$')
plt.imshow(Ivars, origin='lower', extent=[np.min(Lxs), np.max(Lxs), np.min(I0s), np.max(I0s)])
plt.colorbar()

plt.figure()
plt.title(r'$L_X$')
plt.imshow(np.log(Lvars), origin='lower', extent=[np.min(Lxs), np.max(Lxs), np.min(I0s), np.max(I0s)])
plt.colorbar()

plt.show()