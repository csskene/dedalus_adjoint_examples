import logging
from pathlib import Path
import numpy as np
import dedalus.public as d3
import read_vel
import scipy.sparse as sp
from mpi4py import MPI
import dedalus.core.evaluator as evaluator
from dedalus.libraries.dedalus_sphere import zernike

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD

# Parameters
Re = 74345.00585608807
kz = 1
m = 10
Nphi = 2*np.abs(m) + 2
Nr = 128
dtype = np.complex128

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)
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

# Weight matrix
z, weights = zernike.quadrature(2, Nr, k=0)
weights *= 2*np.pi
M = np.sqrt(weights)
Minv = 1/M
# Background (Load in turbulent profile)
w0 = dist.Field(name='w0', bases=disk.radial_basis)
profile = read_vel.get_vel(r)
w0['g'] = profile
# Forcing frequency
omega = dist.Field(name='omega')

# Problem
problem = d3.LBVP([u, w, p, tau_u, tau_w, tau_p], namespace=locals())
problem.add_equation("1j*omega*u + w0*dz(u) + grad(p) - (1/Re)*(lap(u) + dz(dz(u))) + lift(tau_u) = f_disk")
problem.add_equation("1j*omega*w + w0*dz(w) + u@grad(w0) + dz(p) - (1/Re)*(lap(w) + dz(dz(w))) + lift(tau_w) = f_span")
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

# Identify correct m indices
weight_layout = dist.layouts[1]
m_, _ = weight_layout.local_group_arrays(f_disk.domain, scales=1)
m_index = m_==m

# Functions to set fields
def set_field(vec, field_disk, field_span, mat):
    # Sets the field from vec, multiplied by mat 
    vec_split = np.split(np.squeeze(vec), 3)
    field_disk[weight_layout][0, m_index] = mat*vec_split[0]
    field_disk[weight_layout][1, m_index] = mat*vec_split[1]
    field_span[weight_layout][m_index] = mat*vec_split[2]
 
def set_vec(field_disk, field_span, mat):
    return np.hstack([mat*field_disk[weight_layout][0, m_index], 
                          mat*field_disk[weight_layout][1, m_index], 
                          mat*field_span[weight_layout][m_index]])
# Set up resolvent matrix
def mult(vec):
    # Modified resolvent matrix is R_M = (M R Minv)
    # This function multiplies by R_M
    set_field(vec, f_disk, f_span, Minv)
    solver.solve(subproblem)
    matvec = set_vec(u, w, M)
    return matvec

def mult_hermitian(vec):
    # Modified resolvent matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H
    cotangents = {}
    set_field(vec, u_adjoint, w_adjoint, M)
    cotangents[u] = u_adjoint
    cotangents[w] = w_adjoint
    cotangents = solver.compute_sensitivities(cotangents, subproblems=subproblem)
    rmatvec = set_vec(cotangents[f_disk], cotangents[f_span], Minv)
    return rmatvec

# Create scipy linear operator
R = sp.linalg.LinearOperator((3*Nr, 3*Nr), matvec=lambda A: mult(A), 
                             rmatvec=lambda A: mult_hermitian(A))
# Adjoint test
vec1 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
vec2 = np.random.rand(Nr*3) + 1j*np.random.rand(Nr*3)
term1 = np.vdot(vec2, R@vec1)
term2 = np.vdot(R.H@vec2, vec1)
logger.info('Adjoint error = %g' % (np.abs(term1-term2)/np.abs(term1)))

omega_global = np.linspace(0.05, 2, 200)
local_gain1 = []
local_gain2 = []
local_gain3 = []
omega_local = omega_global[comm.rank::comm.size]
for om in omega_local:
    omega['g'] = om
    # Reset scales to 1 and zero fields
    for field in [u, w, f_disk, f_span]:
        field.change_scales(1)
        field['g'] = 0
    # Rebuild solver with different omega
    solver = problem.build_solver()
    # Perform the SVD
    U, sigma, VH = sp.linalg.svds(R, k=3)
    # Set outputs
    # Get forcing from V
    set_field(np.squeeze(np.conj(VH[-1, :]).T), f_disk, f_span, Minv)
    # Get velocity from U
    set_field(np.squeeze(U[:, -1]), u, w, Minv)
    # Set the gains
    local_gain1.append(sigma[-1])
    local_gain2.append(sigma[-2])
    local_gain3.append(sigma[-3])
    # Save the flow data
    scales = (10, 1)
    output_evaluator = evaluator.Evaluator(dist, locals())
    file_name = ('snapshots_omega_{0:.3f}'.format(om)).replace('.', '_')
    data_dir = Path('snapshots', file_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_handler = output_evaluator.add_file_handler(data_dir)
    output_handler.add_task(u, scales=scales)
    output_handler.add_task(w, scales=scales)
    output_handler.add_task(f_disk, scales=scales)
    output_handler.add_task(f_span, scales=scales)
    output_handler.add_task(omega)
    output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)

# Gather outputs
global_outputs = []
for local_output in [local_gain1, local_gain2, local_gain3]:
    local_output = np.array(local_output)
    global_output = np.zeros_like(omega_global)
    global_output[comm.rank::comm.size] = local_output
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, global_output, op=MPI.SUM, root=0)
    else:
        comm.Reduce(global_output, global_output, op=MPI.SUM, root=0)
    global_outputs.append(global_output)

# Save output
if comm.rank==0:
    np.savez('resolvent_gains', omega=omega_global, 
             gain_1=global_outputs[0], 
             gain_2=global_outputs[1],
             gain_3=global_outputs[2])
