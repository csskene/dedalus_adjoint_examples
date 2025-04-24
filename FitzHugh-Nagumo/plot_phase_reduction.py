"""
Plotting script for data generated with phase_reduction.py

Useage:
    python3 plot_phase_reduction.py
"""
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cmap
import numpy as np
import h5py

plt.rcParams.update({
    "text.usetex": True})

if __name__=="__main__":
    cm = 1/2.54  # centimeters in inches
    with h5py.File("FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5") as file:
        # Plot the limit cycle
        u0 = file['tasks/u0'][0].real
        v0 = file['tasks/v0'][0].real
        t = np.squeeze(file['tasks/u'].dims[1][0][:]).real

        # np.savez('phase_npy', u=u0, v=v0, t=t)  # Save LC for calculating trajectory phases
        def plot_coloured_line(data, phases, ax):
            cmap = plt.get_cmap('twilight')
            norm = plt.Normalize(t.min(), t.max())
            points = np.array(data).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(phases)
            lc.set_linewidth(4)
            ax.add_collection(lc)
            return lc

        # Plot the phase sensitivity function
        Zu = file['tasks/Zu'][0].real
        Zv = file['tasks/Zv'][0].real
        norms = file['tasks/norm'][0]
        data_sets = [Zu, Zv]
        ylabels = [r'$Z_u$', r'$Z_v$']

        fig, ax = plt.subplots(figsize=(8.7*cm, 12*cm), nrows=4, gridspec_kw={"height_ratios": [1, 0.1, 1, 1]}, layout='tight')
        cax = ax[1]
        ax0 = ax[0]
        ax1 = ax[2]
        ax2 = ax[3]
        ax2.sharex(ax1)
        
        ax = [ax1, ax2]
        for i, (data, ylabel) in enumerate(zip(data_sets, ylabels)):
            # ax[i].plot(t, data)
            plot_coloured_line([t, data], t, ax[i])
            ax[i].set_ylabel(ylabel)
            ax[i].grid()
            ax[i].set_xlim([0, 2*np.pi])
        ax1.set_ylim([-2.4, 2.4])
        ax2.set_ylim([-7, 7])

        # Plot original LC

        data = np.load('phase_func.npz')
        Psi = data['phase_func']
        U = data['u']
        V = data['v']
        lc = ax0.contourf(V, U, Psi, 100, cmap='twilight', vmin=0, vmax=2*np.pi)
        ax0.plot(v0, u0, color='C1')

        ax0.set_xlim(-0.1, 1.7)
        ax0.set_ylim(-2.2, 2.2)
        ax0.grid()
    
        cbar = plt.colorbar(lc, cax=cax, orientation='horizontal')
        cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

        ax0.set_xlabel(r"$v_0$")
        ax0.set_ylabel(r"$u_0$")
        ax2.set_xlabel(r"$\theta$")
        ax2.set_xticks([0, np.pi/2, np.pi, np.pi*3/2, np.pi*2])
        ax2.set_xticklabels([0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.savefig('FHN_phase_sensitivity.png', dpi=150, bbox_inches='tight')

        






