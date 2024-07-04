"""
Plot Hovmoller diagrams showing the behaviour near the minimal seed
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots(2,figsize=(6, 4), sharex=True)
    files = ['snapshots_E1/snapshots_E1_s1.h5', 'snapshots_E2/snapshots_E2_s1.h5']
    E = [0.2159*0.98,0.2159*1.02]
    for (i,file_name) in enumerate(files):
        with h5py.File(file_name, mode='r') as file:
            u_data = file['tasks/u']
            x = u_data.dims[1][0]
            t = u_data.dims[0][0] - u_data.dims[0][0][0]
            ax[i].pcolormesh(x, t, u_data, cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
            ax[i].set_ylim(0, 50)
            ax[i].set_ylabel('t')
            ax[i].set_title('E0=%f' % (E[i]))
    ax[0].set_xlim(0, 12*np.pi)
    ax[1].set_xlabel('x')
    plt.savefig('Swift_Hohenberg_Hovmoller.png', dpi=200)