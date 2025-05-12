"""
Plot Hovmoller diagrams showing the behaviour near the minimal seed
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    with h5py.File('snapshots/snapshots_s1.h5', mode='r') as file:
        u_task = file['tasks/u']
        x = u_task.dims[1][0]
        t = u_task.dims[0][0]
        t -= t[0]
        plt.pcolormesh(x, t, u_task, cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
    plt.ylim(0, 50)
    plt.ylabel('t')
    plt.xlim(0, 12*np.pi)
    plt.xlabel('x')
    plt.savefig('Swift_Hohenberg_Hovmoller.png', dpi=200)