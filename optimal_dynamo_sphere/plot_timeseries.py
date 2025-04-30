import h5py
import numpy as np
from matplotlib import pyplot as plt
from labellines import labelLines

Rms = [64.3, 64.45, 64.6, 64.8]

plt.rc('text', usetex=True)
plt.rc('font', family='times',size=12)
cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(2, 1, figsize=(8.3*cm, 2*7*cm), layout='tight', sharex=True)

# Plot growth for a range of Rm
xvals = []
for Rm in Rms:
    with h5py.File('high_res/data_Rm_{0:5.03e}/timeseries/timeseries_s1.h5'.format(Rm)) as file:
        A_int = file['tasks/A_int']
        time = A_int.dims[0][0][:] 
        time = time-time[0]
        y = np.log(np.squeeze(A_int)/(4/3*np.pi))
        ax[0].plot(time, y, label=r'$Rm={0:.02f}$'.format(Rm))
        xvals.append(0.7)
labelLines(ax[0].get_lines(), yoffsets=0.015, zorder=2.5, backgroundcolor="none", xvals=xvals, outline_color='none')

linestyles = ['-', '--']
# Compare transient growth
for idx, case in enumerate(['A', 'B']):
    with h5py.File('case_{0:s}_data_Rm_6.500e+01/timeseries/timeseries_s1.h5'.format(case)) as file:
        B_int = file['tasks/B_int']
        time = B_int.dims[0][0][:]
        time = time-time[0]
        y = np.log(np.squeeze(B_int)/(4/3*np.pi))
        y -= y[0]
        ax[1].plot(time, y, label=r'${0:s}$'.format(case), color='C4', linestyle=linestyles[idx])

ax[0].set_ylabel(r'$\log\left(\|\mathbf{A}\|_2/\|\mathbf{A}(0)\|_2\right)$')
y1, y2 = ax[0].get_ylim()
ax[0].set_ylim([0, y2])
ax[0].grid(which='both')
ax[1].grid(which='both')
ax[1].set_xlim([0, 1])
ax[1].set_xlabel(r'$\hat{t}/ (L^2/\eta)$')
ax[1].set_ylabel(r'$\log\left(\|\mathbf{B}\|_2/\|\mathbf{B}(0)\|_2\right)$')
ax[1].legend()

plt.savefig('kinematic_ball_timeseries.eps', dpi=150, bbox_inches='tight')