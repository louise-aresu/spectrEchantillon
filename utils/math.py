import numpy as np
from scipy.special import gamma, factorial, digamma, binom

gen = np.random.default_rng()

def poisson_mass_function(mu, k):
    return np.exp(-mu)*(mu**k)/factorial(k)

def gamma_law(L, size):
    """
    @param:
        - L : Degrees of freedom
        - size : Size of the array

    @return:
        - numpy array (with size 'size')

    Return an array of realization of the gamma law as formalized in
    Multiply stochastic representations for K distributions and  their Poisson transforms
    by Malvin G. Teich and Paul DIament
    """
    return gen.gamma(shape=L, scale=1/L, size=size)


def gamma_mass_function(mu, L, x):
    """
    @param:
        - mu : Mean
        - L : Degrees of freedom
        - x : Real number array

    @return:
        - numpy array (with the same size as x)

    Return an array of the mass function of the gamma law as formalized in
    Multiply stochastic representations for K distributions and  their Poisson transforms
    by Malvin G. Teich and Paul DIament
    """
    return 1/gamma(L)*np.exp(-L*x/mu)*(L/mu)**L * x**(L-1)


def poisson_law(mu, size):
    """
    @param:
        - mu : Mean
        - size : Size of the array
    @return:
        - numpy array (with size 'size')

    Returns an array of realizations of the Poisson law
    """
    return gen.poisson(lam=mu, size=size)


def negbin_mass_function(mu, L, x):
    """
    @param:
        - mu : Mean
        - L : Degrees of freedom
        - x : Integer array
    @return:
        - numpy array (with the same size as x)

    Returns an array of the mass function of the negative binomial law
    """
    return binom(x+L-1, x) * (L / (mu+L))**L * (mu / (mu+L))**x

def negbin_num_moment(mu, L, m):
    res = 0
    for i in range(10):
        res += i**m * negbin_mass_function(mu, L, i)
    return res