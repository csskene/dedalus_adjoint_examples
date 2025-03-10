"""
Plot the neutral curve calculated by running 
neutral_curve.py

Useage:
    python3 neutral_curve.py
"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True})
cm = 1/2.54  # centimeters in inches
neutral = np.load('neutral.npy')

neutral[:, 0] = 10**(neutral[:, 0]*5)
fig = plt.figure(figsize=(8.7*cm, 8.7*cm))
plt.semilogx(neutral[:, 0], neutral[:, 1], 'o-')
plt.xlabel(r'$\textit{Re}$')
plt.ylabel(r'$\alpha$')
plt.grid(which='both')

zoom_x = [5e7, 1e9]
zoom_y = [0.3, 0.35]
plt.plot([zoom_x[0], zoom_x[0], zoom_x[-1], zoom_x[-1], zoom_x[0]],
        [zoom_y[0], zoom_y[-1], zoom_y[-1], zoom_y[0], zoom_y[0]], 'r-')

ax_inset = fig.add_axes([0.55, 0.6, 0.3, 0.2])
neutral = np.load('neutral_zoom.npy')
neutral[:, 0] = 10**(neutral[:, 0]*5)

ax_inset.semilogx(neutral[::2, 0], neutral[::2, 1], 'o-')
ax_inset.set_xlim([5e7, 1e9])
ax_inset.set_ylim([0.3, 0.35])
ax_inset.grid(which='both')

# Set the axis color to red
ax_inset.spines['top'].set_color('red')
ax_inset.spines['bottom'].set_color('red')
ax_inset.spines['left'].set_color('red')
ax_inset.spines['right'].set_color('red')

# Set the major and minor ticks color to red
ax_inset.tick_params(axis='x', which='both', colors='red')
ax_inset.tick_params(axis='y', which='both', colors='red')

# Set the axis font color to red
for label in (ax_inset.get_xticklabels() + ax_inset.get_yticklabels()):
    label.set_color('red')

ax_inset.set_xlabel(r'$\textit{Re}$', color='red')
ax_inset.set_ylabel(r'$\alpha$', color='red')

plt.savefig('poiseuille_neutral.eps', dpi=150, bbox_inches='tight')
