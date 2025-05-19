"""
Plotting script for data generated with phase_reduction.py

Usage:
    python3 plot_phase_reduction.py
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
plt.rcParams.update({"text.usetex": True})


# Setup figure and axes
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(figsize=(8.7*cm, 13*cm), nrows=2, ncols=1, gridspec_kw={"height_ratios": [1, 0.5]})
pax = ax[0]
sax = ax[1]

# Load data
with h5py.File("FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5") as file:
    # Limit cycle
    u0 = file['tasks/u0'][0].real
    v0 = file['tasks/v0'][0].real
    t = np.squeeze(file['tasks/u'].dims[1][0][:]).real
    # Phase sensitivity
    ZU = np.squeeze(file['tasks/Zu']).real
    ZV = np.squeeze(file['tasks/Zv']).real
with np.load('phase_func.npz') as file:
    # Phase function
    Psi = file['phase_func']
    U = file['u']
    V = file['v']

# Plot phase function and limit cycle
lc = pax.pcolormesh(U, V, Psi.T, cmap='twilight', vmin=0, vmax=2*np.pi, shading='gouraud', rasterized=True)
pax.plot(u0, v0, color='w', ls='--', lw=1)
pax.set_ylim(-0.1, 1.7)
pax.set_xlim(-2.2, 2.2)
pax.set_xlabel(r"$u_0$")
pax.set_ylabel(r"$v_0$")

# Colorbar
cbar = plt.colorbar(lc, ax=pax, orientation='horizontal', location='top')
cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
cbar.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
cbar.set_label(r"$\mathrm{Phase}$")

# Sensitivities
sax.hlines(0, 0, 2*np.pi, color='k', ls='--', lw=1)
sax.sharex(cbar.ax)
sax.plot(t, ZU, color='C0', label=r"$z_u$")
sax.plot(t, ZV, color='C1', label=r"$z_v$")
sax.set_ylim([-7, 7])
sax.set_xlabel(r"$\theta$")
sax.legend(loc='lower left')

# Save
plt.tight_layout()
plt.savefig('FHN_phase_sensitivity.pdf', dpi=300)
