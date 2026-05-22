import numpy as np
from scipy.special import gamma, factorial

gen = np.random.default_rng()

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


def gamma_mass_function(L, mu, x):
    """
    @param:
        - L : Degrees of freedom
        - mu : Mean
        - x : Real number array

    @return:
        - numpy array (with the same size as x)

    Return an array of the mass function of the gamma law as formalized in
    Multiply stochastic representations for K distributions and  their Poisson transforms
    by Malvin G. Teich and Paul DIament
    """
    return 1/gamma(L)*(L/mu)**L * x**(L-1)*np.exp(-L*x/mu)


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


def negbin_mass_function(L, mu, x):
    """
    @param:
        - L : Degrees of freedom
        - mu : Mean
        - x : Integer array
    @return:
        - numpy array (with the same size as x)

    Returns an array of the mass function of the negative binomial law
    """
    return L**L / gamma(L) * gamma(x+L)/factorial(x) * mu**x/(mu+L)**(x+L)