import sys
sys.path.append('..')
import time
import matplotlib.pyplot as plt

from sim.pg_distrib import *
from moments import *
from utils.func import *
from sympy import *

def negative_binomial_moments_jacobian():
    mu, L = symbols('mu L')
    m1 = mu
    m2 = mu ** 2 * (1 + 1 / L) + mu

    J  = Matrix([[diff(m1, mu), diff(m1, L)],
                 [diff(m2, mu), diff(m2, L)]])
    return m1, m2, J, mu, L

def iter_pg_distrib_estimation(params, m1e, m2e, sI0=None, sLx=None, steps=3):
    m1, m2, J, mu, L = params

    if sI0 is None:
        sI0 = m1e
    if sLx is None:
        sLx = 5

    r = Matrix([[m1 - m1e], [m2 - m2e]])

    theta = {mu: sI0, L: sLx}
    for _ in range(steps):
        p = J.evalf(subs=theta).LUsolve(-r.evalf(subs=theta))
        # p = -J.evalf(subs=theta)**-1 * r.evalf(subs=theta)
        theta = {mu: theta[mu] + p[0], L: theta[L] + p[1]}
    return theta[mu], theta[L]

def monte_carlo_iter_pg(params, I0, Lx, N, M, gen=None, nbsimu=5, sLx=None, steps=3, pbar=False, hist=False):
    if sLx is None:
        sLx = 5

    vars = np.ndarray((nbsimu, 2))

    if pbar:
        times = np.ndarray(nbsimu)
        for n in range(nbsimu):
            start = time.time()
            _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen)
            m1 = np.mean(y)
            m2 = np.mean(y ** 2)
            vars[n] = iter_pg_distrib_estimation(params, m1, m2, sI0=m1, sLx=sLx, steps=steps)
            end = time.time()
            times[n] = end - start
            progress_bar(n+1, nbsimu, np.mean(times[:n+1]))
    else:
        for n in range(nbsimu):
            _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen)
            m1 = np.mean(y)
            m2 = np.mean(y ** 2)
            vars[n] = iter_pg_distrib_estimation(params, m1, m2, sI0=m1, sLx=sLx, steps=steps)
    mI = np.mean(vars[:, 0])
    mL = np.mean(vars[:, 1])
    vI = np.var(vars[:, 0])
    vL = np.var(vars[:, 1])

    return (mI, vI), (mL, vL)

def robustesse_iter(params, I0, Lx, N, M, Lmin, Lmax, gen=None, steps=3, pbar=False, precision=10, dec=3, nbsimu=10):
    estims = np.ndarray((nbsimu, precision))
    times = np.ndarray(nbsimu * precision)
    Lxe = np.ndarray(nbsimu)
    Lxs = np.linspace(Lmin, Lmax, precision)
    for n in range(nbsimu):
        start = time.time()
        _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen)
        I0e, Lxe[n] = moments_pg_distrib_estimation(y)
        end = time.time()
        times[n*precision] = end - start

        for l in range(precision):
            start = time.time()
            _, estims[n, l] = iter_pg_distrib_estimation(params, np.mean(y), np.mean(y**2), sLx = Lxs[l], steps=steps)
            end = time.time()
            times[precision*n+l] = end - start
            progress_bar(precision*n+l+1, precision*nbsimu, np.mean(times[:precision*n+l+1]))

    plt.figure()
    plt.plot(Lxs, np.mean(estims, axis=0), ':b', marker='+')
    plt.hlines(Lx, Lmin, Lmax, 'k', linestyle=':')
    plt.hlines(np.mean(Lxe), Lmin, Lmax, 'r', linestyle=':')


if __name__ == '__main__':
    #monte_carlo_iter_pg(negative_binomial_moments_jacobian(), 1e-2, 2.6, 128, 7500, nbsimu=20, sLx=3, steps=5, pbar=True)
    robustesse_iter(negative_binomial_moments_jacobian(), 5e-2, 1, 128, 15000, 0.1, 3, pbar=True, precision=10, nbsimu=10)
    plt.show()