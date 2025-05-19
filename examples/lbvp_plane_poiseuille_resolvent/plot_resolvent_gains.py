"""
Plot resolvent gains
Usage:
    python3 plot_resolvent_gains.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Load the data
data = np.load('resolvent_gains.npz')
omega = data['omega']
gains = data['gains']

# Plot the transient growth verus time
fig = plt.figure(figsize=(6, 4))
plt.semilogy(omega, gains, '-.')
plt.ylabel("Gain")
plt.xlabel(r"$\omega$")
plt.title("Optimal gains for plane Poiseuille flow (forced)")
plt.savefig("resolvent_gains.png", dpi=200)
