"""
Routine to get the velocity profiles the following data
(data for Re > 75000 from Princeton superpipe experiments, and for Re = 5000-44000 from Wu and Moin DNS)
Please see the https://github.com/mluhar/resolvent repository for more information and the data.
The data can be obtained here
https://github.com/mluhar/resolvent/blob/modular/allProfilesLogInterp.txt
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def get_vel(r):
    data = np.genfromtxt('allProfilesLogInterp.txt')
    all_Re = np.unique(10**data[:,0])
    # Data at 74345 is the third Re in the file
    data_75000 = data[10**data[:,0]==all_Re[3]] 
    interp_u = interp1d(1-data_75000[:,1], data_75000[:,4])
    return interp_u(r)
