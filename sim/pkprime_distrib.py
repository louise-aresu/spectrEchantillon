import sys
sys.path.append('..')

from utils.math import *
from utils.func import *

def pkprime_distrib(I0, Li, Lx, size, frame):
    """
    @param:
        - I0: Beam intensity
        - Li: Degrees of freedom of the beam Gamma law
        - Lx: Degrees of freedom of the speckle Gamma law
        - size: Dimensions of the sensor in space

    @return:
        - numpy array (with size 'size') of the distribution of the intensity on the sensor
        - numpy array (with size 'size') of photon counts on the sensor
    """
    (N1, N2)    = frame_dim(size)

    ## Simulates the fluctuations of intensity of the beam
    I           = gamma_law(Li, frame)
    I           *= I0

    ## Simulates the speckle
    x           = gamma_law(Lx, (N1, N2, frame))
    x[:, :]     *= I

    ## Simulates the photon counting on the sensor
    y           = poisson_law(x, (N1, N2, frame))

    return x, y