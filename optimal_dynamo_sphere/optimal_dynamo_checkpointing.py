"""
Usage:
    optimal_dynamo.py [options]

Options:
    --Rm=<Rm>                   Magnetic Reynolds number [default: 65]
"""
import logging, sys
from pathlib import Path
import dedalus.public as d3
import numpy as np
from mpi4py import MPI
import scipy.sparse as sp
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.product import Product
from pymanopt.tools.diagnostics import check_gradient
from checkpoint_schedules import SingleMemoryStorageSchedule, HRevolve
import adjoint_tools as tools
from docopt import docopt

args = docopt(__doc__)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank

## Just for now ##
sys.path.append('../manifolds')
from generalized_stiefel import GeneralizedStiefel
##################

from dedalus.extras.flow_tools import GlobalArrayReducer
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

# Parameters
dtype = np.float64
lmax = 15
Nphi = 2*(lmax+1)
Ntheta = lmax+1
Nr = 16
Ro = 1
dealias = 3/2
timestep = 1e-3

Rm = float(args['--Rm'])
if rank==0:
    print('Running with Rm: %f' % Rm)

# Mesh
factors = [[ncpu//i,i] for i in range(1,int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=Ro, dealias=dealias, dtype=dtype)
sphere = ball.surface
phi, theta, r = dist.local_grids(ball)
# Weight matrix
weight_theta = ball.global_colatitude_weights(dist)
weight_r = ball.global_radial_weights(dist)
weight = Ro**3*weight_theta*weight_r*np.ones((Nphi, 1, 1))*(2*np.pi/Nphi)
vol_test = np.sum(weight)
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = ball.volume
weight_layout = dist.layouts[-1]

# Fields
er = dist.VectorField(coords, name='er')
er['g'][2] = 1
omega  = dist.VectorField(coords, name='omega', bases=ball)
u = dist.VectorField(coords, name='u', bases=ball)
tau_u = dist.VectorField(coords, name='tau_u', bases=sphere)
B       = dist.VectorField(coords, name='B', bases=ball)
A       = dist.VectorField(coords, name='A', bases=ball)
tau_A   = dist.VectorField(coords, name='tau_A', bases=sphere)
Pi      = dist.Field(name='Pi', bases=ball)
tau_Pi = dist.Field(name='tau_Pi')

# Boundary conditions
ell_func = lambda ell: ell+1
ellmult = lambda A: d3.SphericalEllProduct(A, coords, ell_func) # mult by (ell+1)
lift = lambda A : d3.Lift(A, ball, -1)

# Problems and solvers
problem_u = d3.LBVP([u, tau_u, Pi, tau_Pi], namespace=locals())
problem_u.add_equation("-lap(u) + grad(Pi) + lift(tau_u) = curl(omega)")
problem_u.add_equation("div(u) + tau_Pi = 0")
problem_u.add_equation("integ(Pi) = 0")
problem_u.add_equation("u(r=Ro) = 0")
solver_u = problem_u.build_solver()

problem_A = d3.LBVP([A, tau_A, Pi, tau_Pi], namespace=locals())
problem_A.add_equation("-lap(A) + grad(Pi) + lift(tau_A) = curl(B)")
problem_A.add_equation("div(A) + tau_Pi = 0")
problem_A.add_equation("integ(Pi) = 0")
problem_A.add_equation("radial(grad(A)(r=Ro)) + ellmult(A)(r=Ro)/Ro = 0")
solver_A = problem_A.build_solver()

problem = d3.IVP([A, u, Pi, tau_Pi, tau_A], namespace=locals())
problem.add_equation("dt(A) - lap(A) + grad(Pi) + lift(tau_A) = Rm*cross(u, curl(A))")
problem.add_equation("dt(u) = 0")
problem.add_equation("div(A) + tau_Pi = 0")
problem.add_equation("integ(Pi) = 0")
problem.add_equation("radial(grad(A)(r=Ro)) + ellmult(A)(r=Ro)/Ro = 0")
solver = problem.build_solver(d3.RK222)

# Cost functional
J = -np.log(d3.integ(B@B))

# Set up direct adjoint looper
total_steps = int(2/timestep)
dal = tools.direct_adjoint_loop(solver, total_steps, timestep, J, pre_solvers=[solver_u, solver_A])

# Set up vectors
global_to_local_omega = tools.global_to_local(weight_layout, omega)
global_to_local_B = tools.global_to_local(weight_layout, B)

N_omega = np.prod(global_to_local_omega.global_shape)
N_B = np.prod(global_to_local_B.global_shape)

grad_omega = np.zeros(N_omega)
grad_B = np.zeros(N_B)

# Set up the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))
manifold_GS_omega = GeneralizedStiefel(N_omega, 1, weight_sp/ball.volume, Binv=weight_inv*ball.volume, retraction="polar")
manifold_GS_B = GeneralizedStiefel(N_B, 1, weight_sp, Binv=weight_inv, retraction="polar")
manifold = Product([manifold_GS_omega, manifold_GS_B])

# Set up checkpointing
# Set up cost and gradient routines
@pymanopt.function.numpy(manifold)
def cost(vec_omega, vec_B):
    global_to_local_omega.vector_to_field(vec_omega, omega)
    global_to_local_B.vector_to_field(vec_B, B)
    dal.reset_initial_condition()
    dal.forward(0, total_steps)
    return dal.functional()

@pymanopt.function.numpy(manifold)
def grad(vec_omega, vec_B):
    global_to_local_omega.vector_to_field(vec_omega, omega)
    global_to_local_B.vector_to_field(vec_B, B)

    dal.reset_initial_condition()
    # Need to recreate the manager each time
    # schedule = SingleMemoryStorageSchedule()
    schedule = HRevolve(total_steps, 400, 50)
    manager = tools.CheckpointingManager(schedule, dal)  # Create the checkpointing manager.
    manager.execute()
    cotangents = dal.gradient()
    global_to_local_omega.field_to_vector(grad_omega, cotangents[omega])
    global_to_local_omega.field_to_vector(grad_B, cotangents[B])
    return [vec.reshape((-1, 1)) for vec in [grad_omega, grad_B]]

# Parallel-safe random point and tangent-vector
random_point = manifold.random_point()
random_point = comm.bcast(random_point, root=0)
random_tangent_vector = manifold.random_tangent_vector(random_point)
random_tangent_vector = comm.bcast(random_tangent_vector, root=0)
# Check the gradient
verbosity = 2*(comm.rank==0)
log_verbosity = 1*(comm.rank==0)
problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-3, log_verbosity=1)
# check_gradient(problem_opt, x=random_point, d=random_tangent_vector)

# A.fill_random(layout='g')
# A.low_pass_filter(scales=0.5)
# omega.fill_random(layout='g')
# omega.low_pass_filter(scales=0.5)
# omega['g'] = d3.curl(omega)['g']
# vec1 = omega.allgather_data(layout=weight_layout).flatten().reshape(-1, 1)
# vec2 = A.allgather_data(layout=weight_layout).flatten().reshape(-1, 1)
# vec1 /= np.sqrt(vec1.T@weight_sp@vec1)
# vec2 /= np.sqrt(vec2.T@weight_sp@vec2)
# initial_point = [vec1, vec2]

# Perform the optimisation
sol = optimizer.run(problem_opt, initial_point=random_point)

# Make data directory
data_dir = Path('data_Rm_{0:5.02e}'.format(Rm))
if rank == 0:  # only do this for the 0th MPI process
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

snapshots = solver.evaluator.add_file_handler(Path("{0:s}/snapshots".format(data_dir)), sim_dt = 0.1)
snapshots.add_task(u, name='u')
snapshots.add_task(omega, name='omega')
snapshots.add_task(B, name='B')

timeseries = solver.evaluator.add_file_handler(Path("{0:s}/timeseries".format(data_dir)), sim_dt = 1e-3)
timeseries.add_task(d3.integ(B@B), name='B_int')
cost(*sol.point)