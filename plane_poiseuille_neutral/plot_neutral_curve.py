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
plt.figure(figsize=(8.7*cm, 8.7*cm))
plt.semilogx(neutral[:, 0], neutral[:, 1], 'o-')
plt.xlabel(r'$\textit{Re}$')
plt.ylabel(r'$\alpha$')
plt.grid(which='both')
plt.savefig('poiseuille_neutral.eps', dpi=150, bbox_inches='tight')
