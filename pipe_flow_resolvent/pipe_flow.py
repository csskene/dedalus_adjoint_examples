import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import read_vel
import scipy.sparse as sp
import dedalus.core.evaluator as evaluator
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

# Forcing frequency
omega = dist.Field(name='omega')
gain_1 = dist.Field(name='gain_1')
gain_2 = dist.Field(name='gain_2')
gain_3 = dist.Field(name='gain_3')

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

# TODO: Fix this hack (find which index is the correct m)
layout = dist.layouts[1]
f_span['g'] = np.exp(m*1j*phi)
mindex = np.argmax(np.linalg.norm(f_span[layout], axis=1))
f_span['g'] = 0
def mult(vec):
    u['g'] = 0
    w['g'] = 0
    # Modified resolvent matrix is R_M = (M R Minv)
    # This function multiplies by R_M 
    vec_split = np.split(np.squeeze(vec), 3)
    f_disk[layout][0][mindex, :] = Minv*vec_split[0]
    f_disk[layout][1][mindex, :] = Minv*vec_split[1]
    f_span[layout][mindex, :] = Minv*vec_split[2]
    solver.solve(subproblem)
    vec_mult = np.hstack([M*u[layout][0][mindex, :], 
                          M*u[layout][1][mindex, :], 
                          M*w[layout][mindex, :]])
    return vec_mult

def mult_hermitian(vec):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H 
    vec_split = np.split(np.squeeze(vec), 3)
    cotangents={}
    u_adjoint[layout][0][mindex, :] = M*vec_split[0]
    u_adjoint[layout][1][mindex, :] = M*vec_split[1]
    w_adjoint[layout][mindex, :] = M*vec_split[2]
    cotangents[u] = u_adjoint
    cotangents[w] = w_adjoint
    cotangents = solver.compute_sensitivities(cotangents, subproblems=subproblem)
    vec_mult_H = np.hstack([Minv*cotangents[f_disk][layout][0][mindex, :], 
                            Minv*cotangents[f_disk][layout][1][mindex, :], 
                            Minv*cotangents[f_span][layout][mindex, :]])
    return vec_mult_H 

# Create scipy linear operator
R = sp.linalg.LinearOperator((3*Nr, 3*Nr), matvec=lambda A: mult(A), 
                             rmatvec=lambda A: mult_hermitian(A))
# Adjoint test
vec1 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
vec2 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
term1 = np.vdot(vec2, R@vec1)
term2 = np.vdot(R.H@vec2, vec1)
logger.info('Adjoint error = %g' % (np.abs(term1-term2)/np.abs(term1)))

# Evaluator for output
scales = (32, 4)
output_evaluator = evaluator.Evaluator(dist, locals())
output_handler = output_evaluator.add_file_handler('snapshots')
output_handler.add_task(u, scales=scales)
output_handler.add_task(w, scales=scales)
output_handler.add_task(f_disk, scales=scales)
output_handler.add_task(f_span, scales=scales)
output_handler.add_task(omega)
output_handler.add_task(gain_1)
output_handler.add_task(gain_2)
output_handler.add_task(gain_3)

omegas = np.linspace(0.05, 2, 70)
forcings = []
for i, om in enumerate(omegas):
    omega['g'] = om
    solver = problem.build_solver()
    # Perform the SVD
    U, sigma, VH = sp.linalg.svds(R, k=3)
    mult(np.conj(VH[-1, :]).T)
    gain_1['g'] = sigma[-1]
    gain_2['g'] = sigma[-2]
    gain_3['g'] = sigma[-3]
    output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=i)
    u.change_scales(1)
    w.change_scales(1)
    f_disk.change_scales(1)
    f_span.change_scales(1)
    forcings.append(VH)
