import numpy as np

def moments_pg_distrib_estimation(x):
    # Calculation of first and second factorial moments
    mf1 = np.mean(x)
    mf2 = np.mean(x*(x-1))
    
    return (mf1, mf1**2 / (mf2 - mf1**2)) # (mu, L)