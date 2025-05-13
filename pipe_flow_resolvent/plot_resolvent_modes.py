"""
Plot resolvent modes
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "text.usetex": True})

filename = 'snapshots/snapshots_s1.h5'
# Plot settings
scale = 1.5
dpi = 300
cm = 1/2.54  # centimeters in inches

fig = plt.figure(layout="constrained", figsize=(8.7*cm, 1.3*8.7*cm))

gs = GridSpec(3, 3, figure=fig)
axgain = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[2, 2])
ax = [ax0, ax1, ax2, ax3, ax4, ax5]

cmap = 'RdBu_r'
# Plot writes
with h5py.File(filename, mode='r') as file:
    axgain.set_xlabel(r'$\omega$')
    axgain.set_ylabel(r'$\sigma$')
    #axgain.grid(True)
    axgain.set_ylim([1, 4e3])
    axgain.set_xlim([0, 1.44])
    omega = file['tasks/omega']
    gain_1 = file['tasks/gain_1']
    gain_2 = file['tasks/gain_2']
    gain_3 = file['tasks/gain_3']
    # Get coords
    dset_u = file['tasks/u']
    dset_w = file['tasks/w']
    dset_f_disk = file['tasks/f_disk']
    dset_f_span = file['tasks/f_span']
    phi = dset_u.dims[2][0][:].reshape(-1, 1)
    r = dset_u.dims[3][0][:].reshape(1, -1)
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    axgain.semilogy(np.squeeze(omega), np.squeeze(gain_1), 'C0', label=r'$\sigma_1$')
    axgain.semilogy(np.squeeze(omega), np.squeeze(gain_2), 'C1', label=r'$\sigma_2$')
    axgain.semilogy(np.squeeze(omega), np.squeeze(gain_3), 'C2', label=r'$\sigma_2$')
    axgain.legend()

    # Get data for the mode with the largest gain
    index = np.argmax(np.squeeze(gain_1))
    u = np.array(dset_u[index])
    w = np.array(dset_w[index])
    f_disk = np.array(dset_f_disk[index])
    f_span = np.array(dset_f_span[index])

    # Fix the phase
    u[:] /= np.exp(1j*10*phi)
    phase_index = np.argmax(np.abs(u[0, 0]))
    phase_factor = u[0, 0, phase_index]
    u[:] /= phase_factor*np.exp(-1j*10*phi)
    w /= phase_factor
    f_disk /= phase_factor
    f_span /= phase_factor

    # Plot the modes
    title_size = 12
    ax[0].pcolormesh(x, y, u[0].real, cmap=cmap, rasterized=True)
    ax[0].set_title(r"$u_\phi$", fontsize=title_size)
    ax[1].pcolormesh(x, y, u[1].real, cmap=cmap, rasterized=True)
    ax[1].set_title(r"$u_r$", fontsize=title_size)
    ax[2].pcolormesh(x, y, w.real, cmap=cmap, rasterized=True)
    ax[2].set_title(r"$u_z$", fontsize=title_size)
    ax[3].pcolormesh(x, y, f_disk[0].real, cmap=cmap, rasterized=True)
    ax[3].set_title(r"$f_\phi$", fontsize=title_size)
    ax[4].pcolormesh(x, y, f_disk[1].real, cmap=cmap, rasterized=True)
    ax[4].set_title(r"$f_r$", fontsize=title_size)
    ax[5].pcolormesh(x, y, f_span.real, cmap=cmap, rasterized=True)
    ax[5].set_title(r"$f_z$", fontsize=title_size)
    for i in range(6):
        theta = np.linspace(0, 2*np.pi, 500)
        ax[i].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        ax[i].set_xlim(-1.05, 1.05)
        ax[i].set_aspect('equal')
        ax[i].set_axis_off()

# Save figure
fig.savefig('pipe_flow.pdf', dpi=dpi)