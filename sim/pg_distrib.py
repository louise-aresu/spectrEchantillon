import numpy as np

import sys
sys.path.append('..')
from utils.func import *
from utils.math import *

def sim_pg_distrib(I0, Lx, size):
    """
    @param:
        - I0: Beam intensity
        - Lx: Degrees of freedom of the speckle Gamma law
        - size: Dimensions of the sensor in space and time

    @return:
        - numpy array (with size 'size') of the distribution of the intensity on the sensor
        - numpy array (with size 'size') of photon counts on the sensor
    """

    ## Simulates the speckle
    x = gamma_law(Lx, size)
    x *= I0

    ## Simulates the photon counting on the sensor
    y = poisson_law(x, size)

    return x, y