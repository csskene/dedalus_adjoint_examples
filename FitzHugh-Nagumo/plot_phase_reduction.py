"""
Plotting script for data generated with phase_reduction.py

Useage:
    python3 plot_phase_reduction.py
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py

if __name__=="__main__":
    cm = 1/2.54  # centimeters in inches
    with h5py.File("FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5") as file:
        # Plot the limit cycle
        u0 = file['tasks/u0'][0]
        v0 = file['tasks/v0'][0]

        fig, ax = plt.subplots(1, 1, figsize=(8.7*cm, 6*cm))
        ax.plot(v0, u0)
        ax.set_xlabel(r"$v_0$")
        ax.set_ylabel(r"$u_0$")
        ax.grid()
        plt.savefig('FHN_limit_cycle.png', dpi=150, bbox_inches='tight')

        # Plot the phase-shift
        u = file['tasks/u'][0]
        v = file['tasks/v'][0]
        t = np.squeeze(file['tasks/u'].dims[1][0][:])
        du0dt = file['tasks/du0dt'][0]
        dv0dt = file['tasks/dv0dt'][0]

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.7*cm, 12*cm))
        ax[0].plot(t, u)
        ax[0].plot(t[::10], du0dt[::10], 'x', label=r'$\frac{du_0}{dt}$')
        ax[0].set_ylabel('u')
        ax[1].plot(t, v)
        ax[1].plot(t[::10], dv0dt[::10], 'x', label=r'$\frac{dv_0}{dt}$')
        ax[1].set_ylabel('v')
        ax[1].set_xlabel(r'$\theta$')
        for i in range(2):
            ax[i].grid()
            ax[i].legend()
            ax[i].set_xlim([0, 2*np.pi])
        ax[1].set_xticks([0, np.pi/2, np.pi, np.pi*3/2, np.pi*2])
        ax[1].set_xticklabels([0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.savefig('FHN_phase_shift.png', dpi=150, bbox_inches='tight')

        # Plot the phase sensitivity function
        Zu = file['tasks/Zu'][0]
        Zv = file['tasks/Zv'][0]
        norms = file['tasks/norm'][0]
        data_sets = [Zu, Zv, norms]
        ylabels = [r'$Z_u$', r'$Z_v$', r'Normalization']
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.7*cm, 12*cm))
        for i, (data, ylabel) in enumerate(zip(data_sets, ylabels)):
            ax[i].plot(t, data)
            ax[i].set_ylabel(ylabel)
            
            ax[i].grid()
            ax[i].set_xlim([0, 2*np.pi])
        ax[2].set_ylim([0.9, 1.1])
        ax[2].set_xlabel(r'$\theta$')
        ax[2].set_xticks([0, np.pi/2, np.pi, np.pi*3/2, np.pi*2])
        ax[2].set_xticklabels([0, r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.savefig('FHN_phase_sensitivity.png', dpi=150, bbox_inches='tight')

        






