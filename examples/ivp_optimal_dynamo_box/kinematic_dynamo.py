"""
Usage:
    kinematic_dynamo.py [options]

Options:
    --Rm=<Rm>                   Magnetic Reynolds number [default: 97.93]
    --test                      Whether to run the Taylor test
"""
import logging
from pathlib import Path
import numpy as np
from dedalus import public as d3
from dedalus.extras.flow_tools import GlobalArrayReducer
import scipy.sparse as sp
from mpi4py import MPI
from docopt import docopt
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.product import Product
from checkpoint_schedules import SingleMemoryStorageSchedule, HRevolve
from adjoint_helper_functions.generalized_stiefel import GeneralizedStiefel
from adjoint_helper_functions import ivp_helpers

args = docopt(__doc__)
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

# Parameters
N = 24
Rm = float(args['--Rm'])
test = args['--test']
dealias = 3/2
timestep = 2.5e-3
timestepper = d3.SBDF2
final_time = 0.5
total_steps = int(final_time/timestep)

logger.info('Running with Rm: %f' % Rm)

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')

# Choose mesh whose factors are most similar in size
factors = [[ncpu//i, i] for i in range(1, int(np.sqrt(ncpu))+1) if np.mod(ncpu, i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 1), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=N, bounds=(0, 1), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=N, bounds=(0, 1), dealias=dealias)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
volume = 1.0

# Fields
phi  = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis))
omega = dist.VectorField(coords, name='omega', bases=(xbasis, ybasis, zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis, ybasis, zbasis))
C = dist.VectorField(coords, name='C')
tau_phi = dist.Field(name='tau_phi')
Pi  = dist.Field(name='Pi', bases=(xbasis, ybasis, zbasis))
tau_Pi = dist.Field(name='tau_Pi')

# Substitutions
B = d3.curl(A)

# Global weight matrix
factor = 1/(2*np.pi) # Length of domain/ (2pi)
weight = factor**3*np.ones((N, N, N))*(np.pi)**3
# Adjust for zeroth rows (multiply each zero mode by 2)
indices = np.indices((N, N, N))
zero_counts = np.sum(indices==0, axis=0)
weight *= 2**zero_counts
# Divide by the volume
weight /= volume
weight_layout = dist.coeff_layout

# Problems
problem = d3.IVP([A, phi, tau_phi, u, C], namespace=locals())
problem.add_equation("dt(A) + grad(phi) - lap(A) + C = Rm*cross(u, B)")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(A) = 0")
problem.add_equation("integ(phi) = 0")
problem.add_equation("dt(u) = 0")
solver = problem.build_solver(timestepper)

problem_u = d3.LBVP([u, Pi, tau_Pi, C], namespace=locals())
problem_u.add_equation("curl(curl(u)) + grad(Pi) + C = curl(omega)")
problem_u.add_equation("div(u) + tau_Pi = 0")
problem_u.add_equation("integ(Pi) = 0")
problem_u.add_equation("integ(u) = 0")
solver_u = problem_u.build_solver()

# Cost functional
J = -np.log(d3.Average(A@A))

# Set up direct adjoint looper
pre_solvers = [solver_u]

dal = ivp_helpers.direct_adjoint_loop(solver, total_steps, timestep, J, adjoint_dependencies=[u, A], pre_solvers=pre_solvers)

# Set up vectors
global_to_local_vec = ivp_helpers.global_to_local(weight_layout, omega)
N_vec = np.prod(global_to_local_vec.global_shape)
grad_omega = np.zeros(N_vec)
grad_mag = np.zeros(N_vec)

# Set up the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))
manifold_GS = GeneralizedStiefel(N_vec, 1, weight_sp, Binv=weight_inv, retraction="polar")
manifold = Product([manifold_GS, manifold_GS])

create_schedule = lambda : SingleMemoryStorageSchedule()
manager = ivp_helpers.CheckpointingManager(create_schedule, dal)  # Create the checkpointing manager.

num_fun_evals = 0
num_grad_evals = 0
# Set up cost and gradient routines
@pymanopt.function.numpy(manifold)
def cost(vec_omega, vec_mag):
    global num_fun_evals
    num_fun_evals += 1
    global_to_local_vec.vector_to_field(vec_omega, omega)
    global_to_local_vec.vector_to_field(vec_mag, A)
    norm = reducer.global_max((d3.integ(omega@omega)/volume)['g'])
    logger.debug('|omega| = %g' % (norm))
    dal.reset_initial_condition()
    manager.execute(mode='forward')
    return dal.functional()

@pymanopt.function.numpy(manifold)
def grad(vec_omega, vec_mag):
    global num_grad_evals
    num_grad_evals += 1
    # Note, cost is always run directly before grad, so no need to recompute
    # the forward pass. Checkpoints are already in manager
    manager.execute(mode='reverse')
    cotangents = dal.gradient()
    global_to_local_vec.field_to_vector(grad_omega, cotangents[omega])
    global_to_local_vec.field_to_vector(grad_mag, cotangents[A])
    return [vec.reshape((-1, 1)) for vec in [grad_omega, grad_mag]]

def random_point(seeds=(None, None)):
    # Parallel-safe random point for omega and B
    # Take curl of random field to ensure
    # omega and B are divergence free
    random_point = []
    for i in range(2):
        omega.fill_random(seed=seeds[i])
        random_field = d3.curl(omega).evaluate()
        norm = reducer.global_max(d3.integ(random_field@random_field)['g'])/volume
        random_field['g'] /= np.sqrt(norm)
        random_field.change_scales(1)
        data = random_field.allgather_data(layout=weight_layout).flatten().reshape((-1, 1))
        random_point.append(data)
    return random_point

# Taylor test
if test:
    point0 = random_point(seeds=(1, 2))
    pointp = random_point(seeds=(3, 4))
    slope, eps_list, residual = ivp_helpers.Taylor_test(cost, grad, point0, pointp)
    logger.info('Result of Taylor test %f' % (slope))
    if rank==0:
        np.savez('box_dynamo_test', eps=np.array(eps_list), residual=np.array(residual))
else:
    # Perform the optimisation
    point = random_point()
    verbosity = 2*(comm.rank==0)
    log_verbosity = 1*(comm.rank==0)
    problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-3, log_verbosity=log_verbosity)
    sol = optimizer.run(problem_opt, initial_point=point)

    logger.info('Number of function evaluations {0:d}'.format(num_fun_evals))
    logger.info('Number of gradient evaluations {0:d}'.format(num_grad_evals))
    # Make data directory
    data_dir = Path('data_Rm_{0:5.03e}'.format(Rm))
    if rank == 0:  # only do this for the 0th MPI process
        data_dir.mkdir(parents=True, exist_ok=True)

    if comm.rank==0:
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]
        np.savez(Path(data_dir, 'convergence'), iterations=iterations, cost_func=costs, residual=gradient_norms)

    snapshots = solver.evaluator.add_file_handler(Path(data_dir, 'snapshots'), sim_dt = 0.1)
    snapshots.add_task(u, name='u')
    snapshots.add_task(omega, name='omega')
    snapshots.add_task(B, name='B')
    snapshots.add_task(A, name='A')
    snapshots.add_task(B@B, name='ME_density')
    snapshots.add_task(u@d3.curl(u), name='helicity')

    timeseries = solver.evaluator.add_file_handler(Path(data_dir, 'timeseries'), sim_dt = 1e-3)
    timeseries.add_task(d3.integ(B@B)/volume, name='B_int')
    timeseries.add_task(d3.integ(A@A)/volume, name='A_int')
    cost(*sol.point)
