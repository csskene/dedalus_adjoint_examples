import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import read_vel
import scipy.sparse as sp
logger = logging.getLogger(__name__)

# Parameters
Re = 74345.00585608807
kz = 1
m = 10
Nphi = 2 * np.abs(m) + 2
Nr = 128
dtype = np.complex128

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
u = dist.VectorField(coords, name='u', bases=disk)
w = dist.Field(name='w', bases=disk)
p = dist.Field(name='p', bases=disk)
tau_u = dist.VectorField(coords, name='tau_u', bases=disk.edge)
tau_w = dist.Field(name='tau_w', bases=disk.edge)
tau_p = dist.Field(name='tau_p')

f_disk = dist.VectorField(coords, name='f_disk', bases=disk)
f_span = dist.Field(name='f_span', bases=disk)

# Substitutions
dz = lambda A: -1j*kz*A
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Weight
# TODO: Something better!
dr = np.diff(r.squeeze())
W = np.zeros(Nr)
W[1:] = dr
W[0] = dr[0]
W = W*r
M = np.sqrt(W)
Minv = 1/M

# Background (Load in turbulent profile)
w0 = dist.Field(name='w0', bases=disk.radial_basis)
w0['g'] = read_vel.get_vel(r)

omega = 0.6
# Problem
problem = d3.LBVP([u, w, p, tau_u, tau_w, tau_p], namespace=locals())
problem.add_equation("1j*omega*u + w0*dz(u) + grad(p) - (1/Re)*(lap(u)+dz(dz(u))) + lift(tau_u) = f_disk")
problem.add_equation("1j*omega*w + w0*dz(w) + u@grad(w0) + dz(p) - (1/Re)*(lap(w)+dz(dz(w))) + lift(tau_w) = f_span")
problem.add_equation("div(u) + dz(w) + tau_p = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("u(r=1) = 0")
problem.add_equation("w(r=1) = 0")
# Solver
solver = problem.build_solver()
subproblem = solver.subproblems_by_group[(m, None)]

# Cotangent fields
u_adjoint = u.copy_adjoint()
w_adjoint = w.copy_adjoint()

# TODO: Fix this hack
layout = dist.layouts[1]
f_span['g'] = np.exp(m*1j*phi)
mindex = np.argmax(np.linalg.norm(f_span[layout], axis=1))
f_span['g'] = 0
def mult(vec):
    # Modified resolvent matrix is R_M = (M R Minv)
    # This function multiplies by R_M 
    vec_split = np.split(np.squeeze(vec), 3)
    f_disk[layout][0][mindex, :] = Minv*vec_split[0]
    f_disk[layout][1][mindex, :] = Minv*vec_split[1]
    f_span[layout][mindex, :] = Minv*vec_split[2]
    solver.solve(subproblem)
    vec_mult = np.hstack([M*u[layout][0][mindex, :], M*u[layout][1][mindex, :], M*w[layout][mindex, :]])
    return vec_mult

def mult_hermitian(vec):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H 
    vec_split = np.split(np.squeeze(vec), 3)
    cotangents={}
    u_adjoint[layout][0][mindex, :] = M*vec_split[0]
    u_adjoint[layout][1][mindex, :] = M*vec_split[1]
    w_adjoint[layout][mindex, :]    = M*vec_split[2]
    cotangents[u] = u_adjoint
    cotangents[w] = w_adjoint
    cotangents = solver.compute_sensitivities(cotangents, subproblems=subproblem)
    vec_mult_H = np.hstack([Minv*cotangents[f_disk][layout][0][mindex, :], Minv*cotangents[f_disk][layout][1][mindex, :], Minv*cotangents[f_span][layout][mindex, :]])
    return vec_mult_H 

# Create scipy linear operator
R = sp.linalg.LinearOperator((3*Nr, 3*Nr),matvec= lambda A: mult(A),rmatvec=lambda A: mult_hermitian(A))
# Adjoint test
vec1 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
vec2 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
term1 = np.vdot(vec2, R@vec1)
term2 = np.vdot(R.H@vec2, vec1)
logger.info('Adjoint error = %g' % (np.abs(term1-term2)/np.abs(term1)))

# Perform the SVD
U, sigma, VH = sp.linalg.svds(R, k=1)
# TODO: Fix this other hack
# Reconstruct the optimal forcing and response
mult(np.conj(VH).T)

# Plot optimal response
scales = (32, 4)
ω = d3.div(d3.skew(u)).evaluate()
ω.change_scales(scales)
u.change_scales(scales)
w.change_scales(scales)
p.change_scales(scales)
phi, r = dist.local_grids(disk, scales=scales)
x, y = coords.cartesian(phi, r)

cmap = 'RdBu_r'
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0,0].pcolormesh(x, y, u['g'][0].real, cmap=cmap)
ax[0,0].set_title(r"$u_\phi$")
ax[0,1].pcolormesh(x, y, u['g'][1].real, cmap=cmap)
ax[0,1].set_title(r"$u_r$")
ax[1,0].pcolormesh(x, y, w['g'].real, cmap=cmap)
ax[1,0].set_title(r"$w$")
ax[1,1].pcolormesh(x, y, p['g'].real, cmap=cmap)
ax[1,1].set_title(r"$p$")
for axi in ax.flatten():
    axi.set_aspect('equal')
    axi.set_axis_off()
fig.tight_layout()
fig.savefig("pipe_output_modes.png", dpi=200)
