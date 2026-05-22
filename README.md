# Photon-counting speckle simulation and estimation using Gamma-Gamma-Poisson distribution (PK')

## Simulator

We simulate the image on the sensor of the bombardment of a sample by a photon beam.

The photon beam is estimated with a Gamma Law of mean $I_0$, the intensity of our beam,
and $L_i$, the degree of freedom.\
The diffraction is estimated with another Gamma Law of mean $\frac{I}{N^2}$ and degree of freedom $L_x$.\
To count photons that hits the sensor we use a Poisson Law of mean the intensity of the diffracted beam.