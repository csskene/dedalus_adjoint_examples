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

# Bases and domain
coords = d3.CartesianCoordinates('y')
dist = d3.Distributor(coords, dtype=dtype, comm=MPI.COMM_SELF)
ybasis = d3.Legendre(coords['y'], size=Ny, dealias=1, bounds=(0, 2))
y, = dist.local_grids(ybasis)

# Fields
u = dist.Field(name='u', bases=ybasis)
v = dist.Field(name='v', bases=ybasis)
w = dist.Field(name='w', bases=ybasis)
p = dist.Field(name='p', bases=ybasis)
tau_u_1 = dist.Field(name='tau_u_1')
tau_u_2 = dist.Field(name='tau_u_2')
tau_v_1 = dist.Field(name='tau_v_1')
tau_v_2 = dist.Field(name='tau_v_2')
tau_w_1 = dist.Field(name='tau_w_1')
tau_w_2 = dist.Field(name='tau_w_2')
tau_p = dist.Field(name='tau_p')

# Substitutions
lift_basis = ybasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)
dy = lambda A: d3.Differentiate(A, coords['y'])

# Base flow
U = dist.Field(name='U', bases=ybasis)
U['g'] = y*(2-y)
Uy = dy(U)

# Problem
problem = d3.IVP([u, v, w, p, tau_u_1, tau_u_2, tau_v_1, tau_v_2, tau_w_1, tau_w_2], namespace=locals())
problem.add_equation("dt(u) + 1j*alpha*u*U + v*Uy - 1/Re*(dy(dy(u)) - alpha**2*u - beta**2*u) + lift(tau_u_1,-1) + lift(tau_u_2,-2) + 1j*alpha*p = 0")
problem.add_equation("dt(v) + 1j*alpha*v*U - 1/Re*(dy(dy(v)) - alpha**2*v - beta**2*v) + lift(tau_v_1,-1) + lift(tau_v_2,-2) + dy(p) = 0")
problem.add_equation("dt(w) + 1j*alpha*w*U - 1/Re*(dy(dy(w)) - alpha**2*w - beta**2*w) + lift(tau_w_1,-1) + lift(tau_w_2,-2) + 1j*beta*p = 0")
problem.add_equation("1j*alpha*u + dy(v) + 1j*beta*w = 0")
# Boundary conditions
problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")
problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")
problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")

# Create solver
solver = problem.build_solver(d3.SBDF2)

# Get spectrally accurate weight matrices
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=ybasis, adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
W = W_field['g']
# Cholesky decomposition
M = np.sqrt(W)
Minv = 1/M
timestep_function = lambda : 0.5

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
        state['g'] = Minv*vec_split[i]
    # Solve the direct problem and return matvec
    solver.evolve(timestep_function, log_cadence=np.inf)
    matvec = np.hstack([M*state['g'] for state in direct_state])
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
        cotangent['g'] = M*vec_split[i]
        cotangents[state] = cotangent
    # Solve the adjoint problem and return rmatvec
    cotangents = solver.compute_sensitivities(cotangents, timestep_function=timestep_function, log_cadence=np.inf) 
    rmatvec = np.hstack([Minv*cotangents[state]['g'] for state in direct_state])
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
global_total_steps = np.array(range(100)[1::5])
local_total_steps = global_total_steps[comm.rank::comm.size]
for total_steps in local_total_steps:
    Phi = create_lin_op(total_steps)
    ts = time.time()
    UH, sigma, V = sp.linalg.svds(Phi, k=1)
    logger.info('T = %f, Time taken for SVD = %f s' % (total_steps*0.5, time.time()-ts))
    local_gains.append(sigma[-1]**2)
    local_times.append(total_steps*0.5)
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
