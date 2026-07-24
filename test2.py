import numpy as np
from sympy import *
from sim.pg_distrib import *
from estim.moments import *
import matplotlib.pyplot as plt
import time
from itertools import product
import multiprocessing as mp

init_printing(use_unicode=True)

mu, L = symbols('mu L')

m1 = mu
m2 = mu ** 2 * (1 + 1 / L) + mu

m1m = diff(m1, mu)
m2m = diff(m2, mu)
m1L = diff(m1, L)
m2L = diff(m2, L)

J = Matrix([[m1m, m1L], [m2m, m2L]])

def gen_vars(I0, Lx, N, M, theta=None, nbsim=10, steps=3):
    if theta is None:
        theta = {mu: 0.001, L: 0.001}
    vals = np.ndarray((nbsim, 2))

    for i in range(nbsim):
        _, y = sim_pg_distrib(I0, Lx, (N, N, M))
        m1e = np.mean(y)
        m2e = np.mean(y ** 2)

        r = Matrix([[m1 - m1e], [m2 - m2e]])

        for _ in range(steps):
            p = J.evalf(subs=theta).LUsolve(-r.evalf(subs=theta))
            # p = -J.evalf(subs=theta)**-1 * r.evalf(subs=theta)
            theta = {mu: theta[mu] + p[0], L: theta[L] + p[1]}
        vals[i] = [theta[mu], theta[L]]

    return np.var(vals[:, 0]), np.var(vals[:, 1])

def estim_iteratif(m1e, m2e, theta, steps):
    r = Matrix([[m1 - m1e], [m2 - m2e]])

    for _ in range(steps):
        p = J.evalf(subs=theta).LUsolve(-r.evalf(subs=theta))
        # p = -J.evalf(subs=theta)**-1 * r.evalf(subs=theta)
        theta = {mu: theta[mu] + p[0], L: theta[L] + p[1]}
    return theta[mu], theta[L]

def histogramme_explicite(I0, Lx, N, M, nbsimu=20, gen=None):
    Ival = np.ndarray(nbsimu)
    Lval = np.ndarray(nbsimu)

    times = np.ndarray(nbsimu)
    for i in range(nbsimu):
        start = time.time()
        _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen)
        end = time.time()
        times[i] = end - start
        progress_bar(i+1, nbsimu, np.mean(times[:i+1]))

        m1 = np.mean(y)
        m2 = np.mean(y ** 2)
        Ival[i] = m1
        Lval[i] = (m1**2) / (m2-m1**2-m1)
    print('')

    plt.figure()
    plt.hist(Ival, density=True, stacked=True, rwidth=0.9)
    plt.figure()
    plt.hist(Lval, density=True, stacked=True, rwidth=0.9)
    print('Valeurs')
    print('I0')
    print('Moyenne: ' f'{np.mean(Ival) : 2e}')
    print('Ecart-type: ' f'{np.std(Ival) : 2e}')
    print('Lx')
    print('Moyenne: ' f'{np.mean(Lval) : 2e}')
    print('Ecart-type: ' f'{np.std(Lval) : 2e}')

def sanity_check(I0, Lx, N, M, steps=10):
    _, y = sim_pg_distrib(I0, Lx, (N, N, M))

    m1e = np.mean(y)
    m2e = np.mean(y ** 2)

    I0expl = m1e
    Lxexpl = m1e**2 / (m2e-m1e**2-m1e)

    theta = {mu: I0expl, L: Lxexpl}
    i0, l = estim_iteratif(y, theta, steps)
    print('Ratio explicite/Newton')
    print('I0: ' f'{I0expl/i0 : e}')
    print('Lx: ' f'{Lxexpl/l : e}')

def resistance_check(I0, Lx, N, M, radius=0.1, resolution=10, steps=5, nbsimu=3):
    maxcounter = resolution**2*nbsimu
    times = np.ndarray(resolution**2*nbsimu)
    vals = np.ndarray((resolution, resolution, nbsimu, 2))
    for n in range(nbsimu):
        _, y = sim_pg_distrib(I0, Lx, (N, N, M))
        m1e = np.mean(y)
        m2e = np.mean(y ** 2)
        r = Matrix([[m1 - m1e], [m2 - m2e]])

        I0expl = m1e
        Lxexpl = m1e**2/(m2e-m1e**2-m1e)

        I0s = np.linspace(I0expl*(1-radius), I0expl*(1+radius), resolution)
        Lxs = np.linspace(Lxexpl*(1-radius), Lxexpl*(1+radius), resolution)

        count = 0
        for i in range(resolution):
            for l in range(resolution):
                theta = {mu: I0s[i], L: Lxs[l]}

                start = time.time()
                vals[i, l, n, :] = estim_iteratif(y, theta, steps)
                end = time.time()
                times[count] = end - start
                count += 1

                progress_bar(count, maxcounter, np.mean(times[:count]))
    print('')

    means = np.ndarray((resolution, resolution, 2))
    vars = np.ndarray((resolution, resolution, 2))
    means[:, :, 0] = np.mean(vals[:, :, :, 0], axis=2)
    means[:, :, 1] = np.mean(vals[:, :, :, 1], axis=2)
    vars[:, :, 0] = np.sqrt(np.var(vals[:, :, :, 0], axis=2))
    vars[:, :, 1] = np.sqrt(np.var(vals[:, :, :, 1], axis=2))

    plt.figure()
    plt.title(r'$I_0$')
    plt.imshow(vars[:, :, 0], origin='lower', aspect='auto',
               extent=(1-radius, 1+radius, 1-radius, 1+radius))
    plt.xlabel(r'Proximité avec $I_0$')
    plt.ylabel(r'Proximité avec $L_X$')

    plt.figure()
    plt.title(r'$L_X$')
    plt.imshow(np.log(vars[:, :, 1]), origin='lower', aspect='auto',
               extent=(1-radius, 1+radius, 1-radius, 1+radius))
    plt.xlabel(r'Proximité avec $I_0$')
    plt.ylabel(r'Proximité avec $L_X$')

def histogramme_iterative(I0, Lx, N, M, theta0=None, nbsimu=20, steps=5, gen=None):
    if theta0 is None:
        theta0 = {mu: 0.005, L: 4}
    vals = np.ndarray((nbsimu, 2))
    times = np.ndarray(nbsimu)
    for n in range(nbsimu):
        start = time.time()
        _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen=gen)
        vals[n, :] = estim_iteratif(y, theta0, steps)
        end = time.time()

        times[n] = end - start
        progress_bar(n+1, nbsimu, np.mean(times[:n+1]))

    plt.figure()
    plt.hist(vals[:, 0], density=True, stacked=True, rwidth=0.9)
    plt.figure()
    plt.hist(vals[:, 1], density=True, stacked=True, rwidth=0.9)

    print('')
    print('Valeurs')
    print('I0')
    print('Moyenne: ' f'{np.mean(vals[:, 0]) : 2e}')
    print('Ecart-type: ' f'{np.std(vals[:, 0]) : 2e}')
    print('Lx')
    print('Moyenne: ' f'{np.mean(vals[:, 1]) : 2e}')
    print('Ecart-type: ' f'{np.std(vals[:, 1]) : 2e}')

def carac_L(I0, Lx, N, M, gen=None, lim=[1, 10], resolution=50, steps=5, nbsimu=20):
    Lvals = np.ndarray((resolution, nbsimu))
    Ls = np.linspace(lim[0], lim[1], resolution)

    times = np.zeros(resolution*nbsimu)

    for n in range(nbsimu):
        start = time.time()
        _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen=gen)
        end = time.time()
        times[n*resolution] = end - start

        for i in range(resolution):
            start = time.time()
            _, Lvals[i, n] = estim_iteratif(y, {mu: np.mean(y), L: Ls[i]}, steps=steps)
            end = time.time()
            times[i+n*resolution] += end - start
            progress_bar(i+1+n*resolution, resolution*nbsimu, np.mean(times[:i+1+n*resolution]))

    plt.figure()
    plt.title(r'$I_0$: ' f'{I0: 2e}')
    plt.errorbar(Ls, np.mean(Lvals, axis=1), 2*np.sqrt(np.var(Lvals, axis=1)),
                marker='+', color='blue', capsize=5)
    plt.ylim((Lx/2, 3/2*Lx))
    plt.xlabel(r'$L_X$ initial')
    plt.ylabel(r'$L_X$ estimé')
    xmin, xmax = plt.xlim()
    plt.hlines(Lx, xmin, xmax, linestyle=':', color='k')

def compar_expl_iter(I0, Lx, N, M, lim=None, steps=3, resolution=30, nbsimu=20, gen=None):
    if lim is None:
        lim = [0.1, 10]
    Lxs = np.linspace(lim[0], lim[1], resolution)
    expl = np.ndarray(nbsimu)
    iter = np.ndarray((nbsimu, resolution))

    times = np.zeros(nbsimu*resolution)
    for n in range(nbsimu):
        start = time.time()
        _, y = sim_pg_distrib(I0, Lx, (N, N, M), gen=gen)
        m1e = np.mean(y)
        m2e = np.mean(y**2)
        expl[n] = m1e**2 / (m2e - m1e**2 - m1e)
        end = time.time()
        times[n*resolution] = end - start
        for l in range(resolution):
            start = time.time()
            _, iter[n, l] = estim_iteratif(m1e, m2e, {mu: np.mean(y), L: Lxs[l]}, steps=steps)
            end = time.time()
            times[n*resolution + l] += end - start
            progress_bar(l+n*resolution + 1, nbsimu*resolution, np.mean(times[:l + n*resolution+1]))
    plt.figure()
    plt.errorbar(Lxs, np.mean(iter, axis=0), 2*np.sqrt(np.var(iter, axis=0)),
                 marker='+', color='blue', capsize=5, label='Iteratif')
    plt.errorbar(Lxs, np.ones_like(Lxs)*np.mean(expl), np.ones_like(Lxs)*2*np.sqrt(np.var(expl)),
                 marker='+', color='red', capsize=5, label='Explicite')
    xmin, xmax = plt.xlim()
    plt.hlines(Lx, xmin, xmax, linestyle=':', color='k')
    plt.legend()

def carac_expl(I0, Lxs, N, M, nbsimu=20, gen=None):
    lenLx = len(Lxs)

    biais = np.ndarray(lenLx)
    variance = np.ndarray(lenLx)
    for l in range(len(Lxs)):
        print(f'Lx: {Lxs[l]}')
        _, (Lxe, Lxv) = monte_carlo_pg(I0, Lxs[l], N, M, nbsimu=nbsimu, gen=gen, log=True)
        biais[l] = (Lxe - Lxs[l])*100/Lxs[l]
        variance[l] = Lxv

    plt.figure()
    plt.plot(Lxs, biais, '--')
    plt.xlabel(r'$L_X$ de simulation')
    plt.ylabel(r'Biais en %')


#seed = np.random.randint(2**32)
#histogramme_explicite(5e-3, 4 ,128, 1500, 200, gen=np.random.default_rng(seed))
#histogramme_iterative(5e-3, 4, 128, 1500, nbsimu=20, gen=np.random.default_rng(seed))
#carac_L(1e-3, 4, 128, 37500, lim=[0.5, 7], steps=3, resolution=15, nbsimu=10, gen=np.random.default_rng(seed))
#carac_L(5e-3, 4, 128, 37500, lim=[0.5, 7], steps=3, resolution=15, nbsimu=10, gen=np.random.default_rng(seed))
#carac_L(1e-2, 4, 128, 37500, lim=[0.5, 7], steps=3, resolution=15, nbsimu=10, gen=np.random.default_rng(seed))
#compar_expl_iter(1e-3, 4, 128, 37500, resolution=20, lim=[0.1, 10], nbsimu=10)
#carac_expl([1e-3, 5e-3, 1e-2, 1e-1], np.arange(1, 11, 1), 128, 100, nbsimu=10)

I0s = [1e-3]
Lxs = np.arange(1, 11)
N = 128
M = 1000
nbsimu = 2
lenI0 = len(I0s)
lenLx = len(Lxs)

biais = np.ndarray((lenI0, lenLx))
std = np.ndarray((lenI0, lenLx))

def sim(args):
    i, l = args
    _, (Lxe, Lxv) = monte_carlo_pg(i, l, N, M, nbsimu, gen=None, log=None, hist=None)
    print(f'Finished process : I {i: .2e} L {l}')
    return [np.abs(l - Lxe)*100/l, np.sqrt(Lxv)]


if __name__ == '__main__':
    # with mp.Pool() as pool:
    #     result = pool.map(sim, [(i, l) for i in I0s for l in Lxs])
    #
    # result = np.array(result).reshape((lenI0, lenLx, 2))
    # biais = result[:, :, 0]
    # std = result[:, :, 1]
    #
    # with open('data_carac_2.npy', 'wb') as f:
    #     np.save(f, np.array([biais, std]))

    # plt.figure()
    # plt.title(r'Biais en fonction de $I_0$ et $L_X$')
    # plt.imshow(biais, aspect='auto', origin="lower",
    #            extent=(np.min(Lxs), np.max(Lxs), np.min(I0s), np.max(I0s)))
    # plt.xlabel(r'$L_X$')
    # plt.ylabel(r'$I_0$')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.title(r'Ecart-type en fonction de $I_0$ et $L_X$')
    # plt.imshow(np.log(variance), aspect='auto', origin="lower",
    #            extent=(np.min(Lxs), np.max(Lxs), np.min(I0s), np.max(I0s)))
    # plt.xlabel(r'$L_X$')
    # plt.ylabel(r'$I_0$')
    # plt.colorbar()
    #
    # plt.show()

    # with open('data_carac_1.npy', 'rb') as f:
    #     biais, std = np.load(f)
    #
    # I, J = np.shape(biais)
    # for j in range(J):
    #     print(f'Lx: {Lxs[j]} => {biais[0, j] : 2.0f} : {std[0, j] : .2e}')

    #_, (Lxe, Lxv) = monte_carlo_pg(1e-2, 6, 128, 7500, gen=None, log=True, hist=False, nbsimu=50)
    #carac_expl(1e-2, np.arange(1, 11), 128, 15000, 20)

    plt.show()