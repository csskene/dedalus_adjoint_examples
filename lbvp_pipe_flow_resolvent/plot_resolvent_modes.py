"""
Plot resolvent modes
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker

plt.rcParams.update({
    "text.usetex": True})

def extrapolate_coordinates(phi, r):
    """Extrapolate data to coordinate boundaries."""
    # Repeat first phi point
    phi_extrap = np.concatenate([phi, [2*np.pi]])
    # Add radial endpoints
    r_extrap = np.concatenate([[0], r, [1]])
    return phi_extrap, r_extrap


def extrapolate_data(phi, r, *data):
    """Extrapolate data to coordinate boundaries."""
    data = np.array(data)
    # Repeat first phi point
    _, r_extrap = extrapolate_coordinates(phi, r)
    # Build data
    shape = np.array(data.shape) + np.array([0, 1, 2])
    data_extrap = np.zeros(shape=shape, dtype=data.dtype)
    # Copy interior data
    data_extrap[:, :-1, 1:-1] = data
    # Copy last point in phi
    data_extrap[:, -1, :] = data_extrap[:, 0, :]
    # Average around origin
    data_extrap[:, :, 0] = np.mean(data_extrap[:, :, 1], axis=(1))[:, None]
    # Extrapolate to outer radius
    data_extrap[:, :, -1] = data_extrap[:, :, -2] + (r_extrap[-1] - r_extrap[-2]) / (r_extrap[-2] - r_extrap[-3]) * (data_extrap[:, :, -2] - data_extrap[:, :, -3])
    return data_extrap

filename = 'snapshots/snapshots_s1.h5'
# Plot settings
scale = 1.5
dpi = 300
cm = 1/2.54  # centimeters in inches

fig = plt.figure(figsize=(8.7*cm, 1.3*8.7*cm))

gs = GridSpec(3, 3, figure=fig, wspace=0)
axgain = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])
ax5 = fig.add_subplot(gs[2, 2])
ax = [ax0, ax1, ax2, ax3, ax4, ax5]

# Plot resolvent gains
with np.load('resolvent_gains.npz') as file:
    omega = file['omega']
    axgain.semilogy(omega, file['gain_1'], 'C0', linestyle='-', label=r'$\sigma_1$')
    axgain.semilogy(omega, file['gain_2'], 'C1', linestyle='-', label=r'$\sigma_2$')
    axgain.semilogy(omega, file['gain_3'], 'C2', linestyle='-', label=r'$\sigma_3$')
    # Find the optimal gain
    omega_max = omega[np.argmax(file['gain_1'])]
axgain.set_xlabel(r'$\omega$')
axgain.set_ylabel(r'$\sigma$')
#axgain.grid(True)
axgain.set_ylim([1, 4e3])
axgain.set_xlim([0, 1.44])
axgain.set_yticks([1, 10, 100, 1000])
axgain.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
axgain.legend()

# Load file corresponding to the maximum gain
filename = ('snapshots_omega_{0:.3f}'.format(omega_max)).replace('.', '_')
filename = 'snapshots/{0:s}/{0:s}_s1.h5'.format(filename)
cmap = 'RdBu_r'
# Plot optimal forcing and response
with h5py.File(filename, mode='r') as file:
    # Get coords
    dset_u = file['tasks/u']
    dset_w = file['tasks/w']
    dset_f_disk = file['tasks/f_disk']
    dset_f_span = file['tasks/f_span']
    phi = dset_u.dims[2][0][:]
    r = dset_u.dims[3][0][:]
    phi_extrap, r_extrap = extrapolate_coordinates(phi, r)

    phi_extrap = phi_extrap.reshape(-1, 1)
    r_extrap = r_extrap.reshape(1, -1)
    x = r_extrap*np.cos(phi_extrap)
    y = r_extrap*np.sin(phi_extrap)

    # Load the fields
    u_phi = np.array(dset_u[0][0])
    u_r = np.array(dset_u[0][1])
    u_z = np.array(dset_w[0])
    u = extrapolate_data(r, phi, u_phi, u_r, u_z)
    f_phi = np.array(dset_f_disk[0][0])
    f_r = np.array(dset_f_disk[0][1])
    f_z = np.array(dset_f_span[0])
    f = extrapolate_data(r, phi, f_phi, f_r, f_z)

    # Plot the modes
    title_size = 12
    flows = [u, f]
    labels = [[r"$u_\phi$", r"$u_r$", r"$u_z$"], [r"$f_\phi$", r"$f_r$", r"$f_z$"]]
    for i, flow in enumerate(flows):
        for j in range(3):
            cplt = ax[3*i + j].contourf(x, y, flow[j].real, levels=100, cmap=cmap)
            # Rasterize
            cplt.set_rasterized(True)
            ax[3*i + j].set_title(labels[i][j], fontsize=title_size)
    
    # Add pipe outline
    for i in range(6):
        theta = np.linspace(0, 2*np.pi, 500)
        ax[i].plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
        ax[i].set_xlim(-1.05, 1.05)
        ax[i].set_ylim(-1.05, 1.05)
        ax[i].set_aspect('equal')
        ax[i].set_axis_off()

    # Move gain axis up to prevent overlapping plots
    pos = axgain.get_position()
    axgain.set_position([pos.x0, pos.y0+0.1, pos.width, pos.height]) # Increase space above ax3

# Save figure
fig.savefig('pipe_flow.pdf', dpi=dpi, bbox_inches='tight')
