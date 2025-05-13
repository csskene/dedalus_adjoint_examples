"""
Plot transient growth
Usage:
    python3 plot_transient_growth.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Load the data
data = np.load('transient_growth.npz')
times = data['times']
gains = data['gains']

# Plot the transient growth verus time
fig = plt.figure(figsize=(6, 4))
plt.semilogy(times, gains,'-.')
plt.ylabel("Gain")
plt.xlabel("T")
plt.title("Optimal gains for plane Poiseuille flow")
plt.savefig("transient_growth.png", dpi=200)
