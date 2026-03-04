"""

"""
import logging, time
import dedalus.public as d3
import numpy as np
from dedalus.tools import jacobi
import scipy.sparse as sp
from mpi4py import MPI
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD

# Parameters
Ny = 128
dtype = np.complex128
alpha = 2
beta = 0
Re = 3000
timestep = 0.1

# Bases and domain
coords = d3.CartesianCoordinates('y')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)
ybasis = d3.Legendre(coords['y'], size=Ny, dealias=3/2, bounds=(0, 2))
y, = dist.local_grids(ybasis)

# Fields
u = dist.Field(name='u', bases=ybasis)
v = dist.Field(name='v', bases=ybasis)
w = dist.Field(name='w', bases=ybasis)
p = dist.Field(name='p', bases=ybasis)
tau_u1 = dist.Field(name='tau_u1')
tau_u2 = dist.Field(name='tau_u2')
tau_v1 = dist.Field(name='tau_v1')
tau_v2 = dist.Field(name='tau_v2')
tau_w1 = dist.Field(name='tau_w1')
tau_w2 = dist.Field(name='tau_w2')
tau_p = dist.Field(name='tau_p')

# Substitutions
lift_basis = ybasis.derivative_basis(1)
dx = lambda A: 1j*alpha*A
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: 1j*beta*A

# First order reduction
lift = lambda A: d3.Lift(A, lift_basis, -1)
# y-derivatives
uy = dy(u) + lift(tau_u1)
vy = dy(v) + lift(tau_v1)
wy = dy(w) + lift(tau_w1)
# Laplacian terms
lap_u = dx(dx(u)) + dy(uy) + dz(dz(u)) + lift(tau_u2)
lap_v = dx(dx(v)) + dy(vy) + dz(dz(v)) + lift(tau_v2)
lap_w = dx(dx(w)) + dy(wy) + dz(dz(w)) + lift(tau_w2)

# Base flow
U = dist.Field(name='U', bases=ybasis)
U['g'] = y*(2-y)

# Problem
problem = d3.IVP([u, v, w, p, tau_u1, tau_u2, tau_v1, tau_v2, tau_w1, tau_w2], namespace=locals())
problem.add_equation("dt(u) - 1/Re*lap_u + dx(p) = -U*dx(u) - v*dy(U)")
problem.add_equation("dt(v) - 1/Re*lap_v + dy(p) = -U*dx(v)")
problem.add_equation("dt(w) - 1/Re*lap_w + dz(p) = -U*dx(w)")
problem.add_equation("dx(u) + vy + dz(w) = 0")
problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")
problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")
problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")

# Create solver
solver = problem.build_solver(d3.RK222)

# Get spectrally accurate weight matrices
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=ybasis, adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
W = W_field['g']
# Cholesky decomposition
M = np.sqrt(W)
Minv = 1/M
timestep_function = lambda : timestep

# Cotangent fields
direct_state = [u, v, w]
adjoint_state = [state.copy_adjoint() for state in direct_state]

# Define direct and hermitian transpose multiplication
def mult(vec, solver, total_steps):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M 
    # Reset solver
    solver.reset()
    solver.stop_iteration = total_steps
    vec_split = np.split(np.squeeze(vec), 3)
    # Initialise the state
    for i, state in enumerate(direct_state):
        state['g', 1] = Minv*vec_split[i]
    # Solve the direct problem and return matvec
    solver.evolve(timestep_function, log_cadence=np.inf)
    matvec = np.hstack([M*state['g', 1] for state in direct_state])
    return matvec

def mult_hermitian(vec, solver, total_steps):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H 
    solver.iteration = total_steps
    solver.timestepper._LHS_params = False
    vec_split = np.split(np.squeeze(vec), 3)
    # Initialise cotangents
    cotangents = {}
    for i, (state, cotangent) in enumerate(zip(direct_state, adjoint_state)):
        cotangent['g', 1] = M*vec_split[i]
        cotangents[state] = cotangent
    # Solve the adjoint problem and return rmatvec
    cotangents = solver.compute_sensitivities(cotangents, timestep_function=timestep_function, log_cadence=np.inf) 
    rmatvec = np.hstack([Minv*cotangents[state]['g', 1] for state in direct_state])
    return rmatvec

# Create linear operator
def create_lin_op(total_steps):
    return sp.linalg.LinearOperator((3*Ny, 3*Ny), matvec=lambda A: mult(A, solver, total_steps), 
                               rmatvec=lambda A: mult_hermitian(A, solver, total_steps))

# Adjoint test on 100 steps
Phi = create_lin_op(100)
vec1 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)
vec2 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)

term2 = np.vdot(Phi.H@vec2, vec1)
term1 = np.vdot(vec2, Phi@vec1)
logger.info('Adjoint error = %g' % np.abs(term1-term2))

# Loop over final times and compute transient growth
ts_tg = time.time()
local_gains = []
local_times = []
global_total_steps = np.array(range(400)[1::20])
local_total_steps = global_total_steps[comm.rank::comm.size]
for total_steps in local_total_steps:
    Phi = create_lin_op(total_steps)
    ts = time.time()
    UH, sigma, V = sp.linalg.svds(Phi, k=1)
    logger.info('T = %f, Time taken for SVD = %f s' % (total_steps*timestep, time.time()-ts))
    local_gains.append(sigma[-1]**2)
    local_times.append(total_steps*timestep)
logger.info('Time taken for whole sweep %f' % (time.time()-ts_tg))

# Gather outputs
global_outputs = []
for local_output in [local_times, local_gains]:
    local_output = np.array(local_output)
    global_output = np.zeros_like(global_total_steps, dtype=np.float64)
    global_output[comm.rank::comm.size] = local_output
    if comm.rank == 0:
        comm.Reduce(MPI.IN_PLACE, global_output, op=MPI.SUM, root=0)
    else:
        comm.Reduce(global_output, global_output, op=MPI.SUM, root=0)
    global_outputs.append(global_output)

# Save output
if comm.rank==0:
    np.savez('transient_growth', times=global_outputs[0], 
             gains=global_outputs[1])
