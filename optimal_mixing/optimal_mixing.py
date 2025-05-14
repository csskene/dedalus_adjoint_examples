"""
This script solves the mixing optimisation problem presented in
'Optimal mixing in two-dimensional stratified plane Poiseuille flow at finite Peclet and Richardson numbers'
F. Marcotte and C.P. Caulfield - JFM 853, 359-385, 2018
Uses Generalised Stiefel manifold to ensure norm-constraint

Usage:
    optimal_mixing.py [options]

Options:
    --test                      Run test only
"""
import numpy as np
from pathlib import Path
from dedalus import public as d3
import logging
from scipy.special import erf
import scipy.sparse as sp
from mpi4py import MPI
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools import jacobi
from scipy.stats import linregress
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from checkpoint_schedules import SingleMemoryStorageSchedule, HRevolve
from docopt import docopt

# TODO: would be nice to remove sys
import sys
sys.path.append('../modules')
from generalized_stiefel import GeneralizedStiefel
import ivp_adjoint_tools as tools

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank

reducer = GlobalArrayReducer(comm)

# Parameters
Nx = 128
Ny = 128
Reinv = 1./500
Peinv = 1./500
Ri = 0.05
dealias = 3/2
T = 5
alpha = 0
timestep = 1e-3
timestepper = d3.SBDF2
total_steps = int(T/timestep)
E0 = 0.02

# Parse arguments
args = docopt(__doc__)
test = args['--test']

# Domain and bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, 4*np.pi), dealias=dealias)
ybasis = d3.Legendre(coords['y'], size=Ny, bounds=(-1, 1), dealias=dealias)
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# Fields and substitutions
u  = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis))
p  = dist.Field(name='p', bases=(xbasis, ybasis))
tau_p  = dist.Field(name='tau_p')
u0 = dist.VectorField(coords, name='u0', bases=(ybasis))
rho  = dist.Field(name='rho', bases=(xbasis, ybasis))
tau_rho1 = dist.Field(name='tau_rho1', bases=(xbasis))
tau_rho2 = dist.Field(name='tau_rho2', bases=(xbasis))
tau_rho  = dist.Field(name='tau_rho', bases=(xbasis))
cost_t  = dist.Field(name='cost_t')

dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

# Fields for the mix norm
psi = dist.Field(name='psi', bases=(xbasis, ybasis))
tau_psi = dist.Field(name='tau_psi') 
tau_psi1 = dist.Field(name='tau_psi1', bases=(xbasis))
tau_psi2 = dist.Field(name='tau_psi2', bases=(xbasis))

# Global weight matrix
# Fourier part
weight = 2*np.ones((Nx,Ny))*np.pi
# Adjust for zeroth rows
weight[0, :] *= 2
# Add Jacobi part
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=(ybasis), adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
weight *= W_field.allgather_data(layout='g').reshape((1, Ny))

# Incorporate energy and domain normalisation
weight *= 0.5/E0/(8*np.pi)
weight_layout = dist.layouts[1]

# Lift and substitutions
lift_basis = ybasis.derivative_basis(1)
lift = lambda A : d3.Lift(A, lift_basis, -1)
lift_basis2 = ybasis.derivative_basis(2)
lift2 = lambda A : d3.Lift(A, lift_basis2, -1)

grad_u = d3.grad(u) + ey*lift(tau_u1)
grad_rho = d3.grad(rho) + ey*lift(tau_rho1)
grad_psi = d3.grad(psi) + ey*lift(tau_psi1)

# Base-flow
u0['g'][0] = 1-y**2

# Problems
problem = d3.IVP([u, rho, p, cost_t, tau_u1, tau_u2, tau_rho1, tau_rho2, tau_p], namespace=locals())
problem.add_equation("dt(u) - Reinv*div(grad_u) + grad(p) + lift2(tau_u2) + Ri*ey*rho + u0@grad(u) + u@grad(u0) = -u@grad(u)")
problem.add_equation("dt(rho) - Peinv*div(grad_rho) + lift2(tau_rho2) + u0@grad(rho) = -div(u*rho)")
problem.add_equation("dt(cost_t) = integ(u@u)")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("u(y=-1) = 0")
problem.add_equation("u(y=1) = 0")
problem.add_equation("dy(rho)(y=-1) = 0")
problem.add_equation("dy(rho)(y=1)  = 0")
solver = problem.build_solver(timestepper)

problem_mix_norm = d3.LBVP([psi, tau_psi1, tau_psi2, tau_psi], namespace=locals())
problem_mix_norm.add_equation("div(grad_psi) + lift(tau_psi2) + tau_psi = rho")
problem_mix_norm.add_equation("dy(psi)(y=-1) = 0")
problem_mix_norm.add_equation("dy(psi)(y=1) = 0")
problem_mix_norm.add_equation("integ(psi) = 0")
solver_psi = problem_mix_norm.build_solver()

# Cost functional
J = (alpha/2*d3.integ(d3.grad(psi)@d3.grad(psi)) - (1-alpha)/(2*T)*cost_t)

# Set up direct adjoint loop
post_solvers = [solver_psi]
dal = tools.direct_adjoint_loop(solver, total_steps, timestep, J, adjoint_dependencies=[u, rho], post_solvers=post_solvers)

# Set up vectors
global_to_local_vec = tools.global_to_local(weight_layout, u)
N_vec = np.prod(global_to_local_vec.global_shape)
grad_u = np.zeros(N_vec)

# Set up checkpointing
create_schedule = lambda : SingleMemoryStorageSchedule()
manager = tools.CheckpointingManager(create_schedule, dal)  # Create the checkpointing manager.

# Set up manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten()]))
manifold = GeneralizedStiefel(N_vec, 1, weight_sp, Binv=weight_inv, retraction="polar")

num_fun_evals = 0
num_grad_evals = 0
# Set up cost and gradient routines
@pymanopt.function.numpy(manifold)
def cost(vec_u):
    global num_fun_evals
    num_fun_evals += 1
    global_to_local_vec.vector_to_field(vec_u, u)
    norm = reducer.global_max((0.5*d3.integ(u@u)/(8*np.pi))['g'])
    logger.debug('|u| = %g' % (norm))
    rho.change_scales(1)
    rho['g'] = -0.5*erf(y/0.125)
    cost_t['g'] = 0
    dal.reset_initial_condition()
    manager.execute(mode='forward')
    return dal.functional()

@pymanopt.function.numpy(manifold)
def grad(vec_u):
    global num_grad_evals
    num_grad_evals += 1
    # Note, cost is always run directly before grad, so no need to recompute
    # the forward pass. Checkpoints are already in manager
    manager.execute(mode='reverse')
    cotangents = dal.gradient()
    global_to_local_vec.field_to_vector(grad_u, cotangents[u])
    return grad_u.reshape((-1, 1))

phi = dist.Field(name='phi', bases=(xbasis, ybasis)) # Streamfunction
def random_point():
    # Parallel-safe random point for u
    phi.fill_random(layout='g')
    phi.low_pass_filter(scales=0.6)
    u.change_scales(dealias)
    u['g'][0] = dy(phi)['g']
    u['g'][1] = -dx(phi)['g']
    u.change_scales(1)
    data = u.allgather_data(layout=weight_layout).flatten().reshape((-1, 1))
    data /= np.sqrt(np.vdot(data, weight_sp@data))
    return data.reshape((-1, 1))

if test:
    point_0 = random_point()
    point_p = random_point()
    residual = []
    cost_0 = cost(point_0)
    grad_0 = grad(point_0)
    dJ = np.vdot(grad_0, point_p)
    eps = 1e-4
    eps_list = []
    for i in range(10):
        eps_list.append(eps)
        point = point_0 + eps*point_p
        cost_p = cost(point)
        residual.append(np.abs(cost_p - cost_0 - eps*dJ))
        eps /= 2
    regression = linregress(np.log(eps_list), y=np.log(residual))
    logger.info('Result of Taylor test %f' % (regression.slope))
    np.savez('mixing_test', eps=np.array(eps_list), residual=np.array(residual))
else:
    # Perform the optimisation
    verbosity = 2*(comm.rank==0)
    log_verbosity = 1*(comm.rank==0)
    problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-2, log_verbosity=1)
    initial_point = random_point()
    sol = optimizer.run(problem_opt, initial_point=initial_point)

    logger.info('Number of function evaluations {0:d}'.format(num_fun_evals))
    logger.info('Number of gradient evaluations {0:d}'.format(num_grad_evals))

    # Make data directory
    data_dir = Path('data')
    if rank == 0:  # only do this for the 0th MPI process
        if not data_dir.exists():
            data_dir.mkdir(parents=True)

    # Save outputs
    if comm.rank==0:
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]
        np.savez('convergence', iterations=iterations, costs=costs, gradient_norms=gradient_norms)

    snapshots = solver.evaluator.add_file_handler(Path(data_dir, 'snapshots'), sim_dt = 0.1, mode='overwrite')
    vorticity = -d3.div(d3.skew(u))
    snapshots.add_task(u)
    snapshots.add_task(vorticity, name='vorticity')
    snapshots.add_task(rho)

    timeseries = solver.evaluator.add_file_handler(Path(data_dir, 'timeseries'), sim_dt = 5e-3, mode='overwrite')
    timeseries.add_task(0.5*d3.integ(u@u)/(8*np.pi), name='KE')

    cost(sol.point)