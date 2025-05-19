import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='times', size=12)

cm = 1/2.54  # centimeters in inches
data = np.load('Lane_Emden.npz')

plt.figure(figsize=(8.7*cm, 6*cm))
plt.plot(data['n'], data['R'], '-')
# Plot solid lines showing the gradients
gap = 0.01
for i, n in enumerate(data['n']):
    p = data['R'][i] + gap*data['dR'][i]
    m = data['R'][i] - gap*data['dR'][i] 
    plt.plot([data['n'][i]-gap, data['n'][i]+gap], [m, p], color='black', linewidth=2)

plt.xlabel(r'$n$')
plt.ylabel(r'$X$')
plt.grid()
plt.savefig('lane_emden_adaptive.pdf', bbox_inches='tight')
