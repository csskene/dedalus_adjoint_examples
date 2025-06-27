import h5py
import numpy as np
from matplotlib import pyplot as plt
from labellines import labelLines

Rms = [64.3, 64.45, 64.6, 64.8]

plt.rc('text', usetex=True)
plt.rc('font', family='times',size=12)
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(2, 1, figsize=(7*cm, 10*cm), layout='tight', sharex=True)

# Plot growth for a range of Rm
xvals = []
for Rm in Rms:
    with h5py.File('case_A_data_Rm_{0:5.03e}/timeseries/timeseries_s1.h5'.format(Rm)) as file:
        A_int = file['tasks/A_int']
        time = A_int.dims[0][0][:]
        time = time-time[0]
        y = np.log(np.squeeze(A_int))
        ax[0].plot(time, y, label=rf'$\mathrm{{Rm}}={Rm:.02f}$')
        xvals.append(0.73)
labelLines(ax[0].get_lines(), yoffsets=0.025, zorder=2.5, backgroundcolor="none", xvals=xvals, outline_color='none', fontsize=9)

linestyles = ['-', '--']
# Compare transient growth
for idx, case in enumerate(['A', 'B']):
    with h5py.File('case_{0:s}_data_Rm_6.480e+01/timeseries/timeseries_s1.h5'.format(case)) as file:
        B_int = file['tasks/B_int']
        time = B_int.dims[0][0][:]
        time = time-time[0]
        y = np.log(np.squeeze(B_int))
        y -= y[0]
        ax[1].plot(time, y, label=rf"Optimize $\|\mathbf{{{case}}}\|^2$", color='C3', linestyle=linestyles[idx],)
labelLines(ax[1].get_lines()[0:1], yoffsets=0.11, zorder=2.5, backgroundcolor="none", xvals=[0.5], outline_color='none', fontsize=10)
labelLines(ax[1].get_lines()[1:2], yoffsets=-0.13, zorder=2.5, backgroundcolor="none", xvals=[0.5], outline_color='none', fontsize=10)

ax[0].set_ylabel(r'$\log\left(\|\mathbf{A}\|^2/\|\mathbf{A}(0)\|^2\right)$')
ax[0].set_ylim([0, None])
ax[1].set_xlim([0, 1])
ax[1].set_xlabel(r'$\hat{t}/ (L^2/\eta)$')
ax[1].set_ylabel(r'$\log\left(\|\mathbf{B}\|^2/\|\mathbf{B}(0)\|^2\right)$')

plt.savefig('kinematic_ball_timeseries.pdf', bbox_inches='tight')