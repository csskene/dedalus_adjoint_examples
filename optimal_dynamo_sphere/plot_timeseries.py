import h5py
import numpy as np
from matplotlib import pyplot as plt
from labellines import labelLines

Rms = [64.3, 64.45, 64.6, 64.8]

plt.rc('text', usetex=True)
plt.rc('font', family='times',size=12)
cm = 1/2.54  # centimeters in inches
plt.figure(figsize=(8.3*cm, 7*cm))
xvals = []
for Rm in Rms:  
    data = h5py.File('high_res/data_Rm_{0:5.03e}/timeseries/timeseries_s1.h5'.format(Rm))
    A_int = data['tasks/A_int']
    time = A_int.dims[0][0][:] 
    time = time-time[0]
    y = np.log(np.squeeze(A_int)/(4/3*np.pi))
    plt.plot(time, y, label=r'$Rm={0:.02f}$'.format(Rm))
    xvals.append(0.7)

labelLines(plt.gca().get_lines(), yoffsets=0.015, zorder=2.5, backgroundcolor="none", xvals=xvals, outline_color='none')
plt.xlabel(r'$\hat{t}/ (L^2/\eta)$')
plt.ylabel(r'$\log\left(\frac{1}{V}\int \mathbf{A}\cdot\mathbf{A} \;\textrm{d}V\right)$')
plt.xlim([0, 1])
y1, y2 = plt.gca().get_ylim()
plt.ylim([0, y2])
plt.grid(which='both')
plt.savefig('kinematic_ball_timeseries.eps', dpi=150, bbox_inches='tight')