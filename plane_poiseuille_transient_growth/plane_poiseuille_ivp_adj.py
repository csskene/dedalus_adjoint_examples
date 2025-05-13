"""

"""
import logging, time
import dedalus.public as d3
import numpy as np
from dedalus.tools import jacobi
import scipy.sparse as sp
logger = logging.getLogger(__name__)

# Parameters
Ny = 128
dtype = np.complex128
alpha = 2
beta = 0
Re = 3000

# Bases and domain
coords = d3.CartesianCoordinates('y')
dist = d3.Distributor(coords, dtype=dtype)
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
term2 = np.vdot(Phi.H@vec2,vec1)
term1 = np.vdot(vec2, Phi@vec1)
term2 = np.vdot(Phi.H@vec2,vec1)
term1 = np.vdot(vec2, Phi@vec1)
logger.info('Adjoint error = %g' % np.abs(term1-term2))

# Loop over final times and compute transient growth
ts_tg = time.time()
gains = []
times = []
for total_steps in range(100)[1::5]:
    Phi = create_lin_op(total_steps)
    ts = time.time()
    UH, sigma, V = sp.linalg.svds(Phi, k=1)
    logger.info('T = %f, Time taken for SVD = %f s' % (total_steps*0.5, time.time()-ts))
    gains.append(sigma[-1]**2)
    times.append(total_steps*0.5)
logger.info('Time taken for whole sweep %f' % (time.time()-ts_tg))

# Save output
np.savez('transient_growth', times=times, gains=gains)