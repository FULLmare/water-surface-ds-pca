"""Intensity conversion and Gaussian-kernel utilities for DS+PCA."""
import numpy as np

KEY = ["wind_speed_mps", "frequency_GHz", "theta_i_deg", "polarization"]


def db(x):
    return 10*np.log10(np.maximum(x,1e-8))




def kernel_features(c):
    return np.column_stack([c.wind_speed_mps.to_numpy()/3,
                            np.log(c.frequency_GHz.to_numpy())/np.log(2),
                            np.cos(np.deg2rad(c.theta_i_deg.to_numpy()))/0.3,
                            c.polarization.eq('TM').to_numpy()*4])


def kernel(x,y):
    d=np.sum((x[:,None,:]-y[None,:,:])**2,axis=-1)
    return np.exp(-d/2)
