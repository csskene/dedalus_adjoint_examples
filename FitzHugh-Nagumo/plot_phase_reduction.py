"""
Plotting script for data generated with phase_reduction.py

Useage:
    python3 plot_phase_reduction.py
"""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import h5py

plt.rcParams.update({
    "text.usetex": True})

if __name__=="__main__":
    # Setup figure and axes
    cm = 1/2.54  # centimeters in inches
    fig, ax = plt.subplots(figsize=(8.7*cm, 12*cm), nrows=3, ncols=1, gridspec_kw={"height_ratios": [1, 0.1, 1]}, layout='tight')
    ax0 = ax[0]
    cax = ax[1]
    ax1 = ax[2]
    ax1.sharex(cax)
    ax1.set_ylim([-2.4, 2.4])
    ax2 = ax1.twinx()

    # Get data
    with h5py.File("FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5") as file:
        # Get limit cycle data
        u0 = file['tasks/u0'][0].real
        v0 = file['tasks/v0'][0].real
        t = np.squeeze(file['tasks/u'].dims[1][0][:]).real
        # Get phase sensitivity data
        ZU = np.squeeze(file['tasks/Zu']).real
        ZV = np.squeeze(file['tasks/Zv']).real

    # Get phase function data
    with np.load('phase_func.npz') as data:
        Psi = data['phase_func']
        U = data['u']
        V = data['v']
    
    # Plot the limit cycle and phase function
    lc = ax0.pcolormesh(V, U, Psi, cmap='twilight', vmin=0, vmax=2*np.pi, shading='gouraud')
    ax0.plot(v0, u0, color='C2')
    ax0.set_xlim(-0.1, 1.7)
    ax0.set_ylim(-2.2, 2.2)
    ax0.grid()
    # ax0.set_aspect('equal')
    ax0.set_xlabel(r"$v_0$")
    ax0.set_ylabel(r"$u_0$")

    # Display the colorbar
    cbar = plt.colorbar(lc, cax=cax, orientation='horizontal')
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    ax1.plot(t, ZU, color='C0')
    ax2.plot(t, ZV, color='C1')
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$z_u$", color='C0')
    ax2.set_ylabel(r"$z_v$", color='C1')
    ax1.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, np.pi*2])
    ax1.set_xticklabels([0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax1.grid()
    # Change color of yaxis to match lines
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.yaxis.set_tick_params(color='C0')
    ax2.yaxis.set_tick_params(color='C1')
    ax2.spines['left'].set_color('C0')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.spines['right'].set_color('C1')
    # Save the figure
    plt.savefig('FHN_phase_sensitivity.png', dpi=150, bbox_inches='tight')
