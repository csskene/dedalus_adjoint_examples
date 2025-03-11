"""
This script optimises kinematic dynamo action in a ball by optimising the velocity
field and initial magnetic field [1]. The initial velocity field is constrained
to have unit enstrophy. We solve for the vector potential A under the Coulomb gauge
(B = curl(A), div(A)=0), and so we optimise the growth of A, subject to it having
an initial norm of 1.

Checkpointing is provided using the checkpoint_schedules library.
Norm constraints are handled using pymanopt.

[1] The optimal kinematic dynamo driven by steady flows in a sphere,
    Chen L., Herreman W., Li K., Livermore P.W., Luo J.W., Jackson A.,
    Journal of Fluid Mecahnics, 839:1-32, 2018

Usage:
    optimal_dynamo.py [options]

Options:
    --Rm=<Rm>                   Magnetic Reynolds number [default: 65]
    --T=<T>                     Simulation stop time [default: 1]
    --case=<case>               Optimise for A, or B0 [default: A]
    --test                      Whether to run the Taylor test
    --checkpoint                Whether to use checkpoints
    --N_ram=<N_ram>             Number of checkpoints in ram [default: 400]
    --N_disk=<N_disk>           Number of checkpoints on disk [default: 50]
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
from checkpoint_schedules import SingleMemoryStorageSchedule, HRevolve
import adjoint_tools as tools
from scipy.stats import linregress
from docopt import docopt

args = docopt(__doc__)
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank

# TODO: would be nice to remove sys
sys.path.append('../manifolds')
from generalized_stiefel import GeneralizedStiefel

from dedalus.extras.flow_tools import GlobalArrayReducer
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

# Parameters
dtype = np.float64
lmax = 15
Nphi = 2*(lmax+1)
Ntheta = lmax+1
Nr = 32
Ro = 1
dealias = 3/2
timestep = 5e-4

Rm = float(args['--Rm'])
T = float(args['--T'])
case = args['--case']
logger.info('Running with Rm: %f' % Rm)
logger.info('Running with T: %f' % T)
logger.info('Optimisation case: %s' % case)
test = args['--test']
checkpoint = args['--checkpoint']

# Mesh
factors = [[ncpu//i,i] for i in range(1, int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
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
omega = dist.VectorField(coords, name='omega', bases=ball)
u = dist.VectorField(coords, name='u', bases=ball)
tau_u = dist.VectorField(coords, name='tau_u', bases=sphere)
B0 = dist.VectorField(coords, name='B0', bases=ball)
A = dist.VectorField(coords, name='A', bases=ball)
tau_A = dist.VectorField(coords, name='tau_A', bases=sphere)
Pi = dist.Field(name='Pi', bases=ball)
tau_Pi = dist.Field(name='tau_Pi')

# Substitutions
if case=='A' or case=='B0':
    B = d3.curl(A)
elif case=='B':
    B = dist.VectorField(coords, name='B', bases=ball)

# Boundary conditions
ell_func = lambda ell: ell + 1
ellmult = lambda A: d3.SphericalEllProduct(A, coords, ell_func) # mult by (ell+1)
lift = lambda A : d3.Lift(A, ball, -1)

# Problems
problem_u = d3.LBVP([u, tau_u, Pi, tau_Pi], namespace=locals())
problem_u.add_equation("curl(curl(u)) + grad(Pi) + lift(tau_u) = curl(omega)")
problem_u.add_equation("div(u) + tau_Pi = 0")
problem_u.add_equation("integ(Pi) = 0")
problem_u.add_equation("u(r=Ro) = 0")
solver_u = problem_u.build_solver()

problem_A = d3.LBVP([A, tau_A, Pi, tau_Pi], namespace=locals())
problem_A.add_equation("curl(curl(A)) + grad(Pi) + lift(tau_A) = curl(B0)")
problem_A.add_equation("div(A) + tau_Pi = 0")
problem_A.add_equation("integ(Pi) = 0")
problem_A.add_equation("radial(grad(A)(r=Ro)) + ellmult(A)(r=Ro)/Ro = 0")
solver_A = problem_A.build_solver()

if case=='A' or case=='B0':
    problem = d3.IVP([A, u, Pi, tau_Pi, tau_A], namespace=locals())
    problem.add_equation("dt(A) - lap(A) + grad(Pi) + lift(tau_A) = Rm*cross(u, B)")
    problem.add_equation("dt(u) = 0")
    problem.add_equation("div(A) + tau_Pi = 0")
    problem.add_equation("integ(Pi) = 0")
    problem.add_equation("radial(grad(A)(r=Ro)) + ellmult(A)(r=Ro)/Ro = 0")
elif case=='B':
    problem = d3.IVP([B, u, Pi, tau_A], namespace=locals())
    problem.add_equation("dt(B) - lap(B) + grad(Pi) + lift(tau_A) = Rm*curl(cross(u, B))")
    problem.add_equation("dt(u) = 0")
    problem.add_equation("div(B) = 0")
    problem.add_equation("radial(radial(grad(B)(r=Ro)) + (ellmult(B)/Ro)(r=Ro)) = 0", condition="ntheta!=0")
    problem.add_equation("radial(curl(B)(r=Ro)) = 0", condition="ntheta!=0")
    problem.add_equation("Pi(r=Ro) = 0")
solver = problem.build_solver(d3.SBDF2)

# Cost functional
if case=='A':
    J = -np.log(d3.integ(A@A))
elif case=='B0' or case=='B':
    J = -np.log(d3.integ(B@B))

# Set up direct adjoint looper
total_steps = int(T/timestep)

pre_solvers = [solver_u]
if case=="B0":
    pre_solvers += [solver_A]
dal = tools.direct_adjoint_loop(solver, total_steps, timestep, J, adjoint_dependencies=[A], pre_solvers=pre_solvers)

# Set up vectors
global_to_local_vec = tools.global_to_local(weight_layout, omega)
N_vec = np.prod(global_to_local_vec.global_shape)
grad_omega = np.zeros(N_vec)
grad_mag = np.zeros(N_vec)

# Set up the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))
manifold_GS = GeneralizedStiefel(N_vec, 1, weight_sp/ball.volume, Binv=weight_inv*ball.volume, retraction="polar")
manifold = Product([manifold_GS, manifold_GS])

# Set up checkpointing
if checkpoint:
    N_ram = int(args['--N_ram'])
    N_disk = int(args['--N_disk'])
    create_schedule = lambda : HRevolve(total_steps, N_ram, N_disk)
    logger.info('Checkpointing with N_ram={0:d}, N_disk={1:d}'.format(N_ram, N_disk))
else:
    create_schedule = lambda : SingleMemoryStorageSchedule()
manager = tools.CheckpointingManager(create_schedule, dal)  # Create the checkpointing manager.

# Set up cost and gradient routines
@pymanopt.function.numpy(manifold)
def cost(vec_omega, vec_mag):
    global_to_local_vec.vector_to_field(vec_omega, omega)
    if case=='A':
        global_to_local_vec.vector_to_field(vec_mag, A)
    elif case=='B0':
        global_to_local_vec.vector_to_field(vec_mag, B0)
    elif case=='B':
        global_to_local_vec.vector_to_field(vec_mag, B)
    ## For debugging ###########################################
    init_norm = reducer.global_max(d3.integ(omega@omega)['g'])/ball.volume
    logger.debug('|omega|-1 = %g' % (init_norm-1))
    solver_u.solve()
    missfit = d3.curl(u)-omega
    norm = reducer.global_max(d3.integ(missfit@missfit)['g'])/reducer.global_max(d3.integ(u@u)['g'])
    logger.debug('|curl(u) - omega| = %g' % (norm))
    norm = reducer.global_max(np.abs(tau_u['g']))
    logger.debug('max(tau_u) = %g' % (norm))
    if case=='B0':
        solver_A.solve()
        init_norm = reducer.global_max(d3.integ(B@B)['g'])/ball.volume
        logger.debug('|B|-1 = %g' % (init_norm-1))
        missfit = B0-d3.curl(A)
        norm = reducer.global_max(d3.integ(missfit@missfit)['g'])/reducer.global_max(d3.integ(B0@B0)['g'])
        logger.debug('|curl(A) - B0| = %g' % (norm))
        norm = reducer.global_max(np.abs(tau_A['g']))
        logger.debug('max(tau_A) = %g' % (norm))
    #############################################################
    dal.reset_initial_condition()
    manager.execute(mode='forward')
    return dal.functional()

@pymanopt.function.numpy(manifold)
def grad(vec_omega, vec_mag):
    # Note, cost is always run directly before grad, so no need to recompute
    # the forward pass. Checkpoints are already in manager
    manager.execute(mode='reverse')
    cotangents = dal.gradient()
    global_to_local_vec.field_to_vector(grad_omega, cotangents[omega])
    if case=='A':
        global_to_local_vec.field_to_vector(grad_mag, cotangents[A])
    elif case=='B0':
        global_to_local_vec.field_to_vector(grad_mag, cotangents[B0])
    elif case=='B':
        global_to_local_vec.field_to_vector(grad_mag, cotangents[B])
    return [vec.reshape((-1, 1)) for vec in [grad_omega, grad_mag]]

def random_point():
    # Parallel-safe random point for omega and B
    # Use LBVPs to create initial guesses which are divergence
    # free and satisfy the correct boundary conditions
    random_point = []
    omega.fill_random('g')
    omega.low_pass_filter(scales=0.25);omega['c'];omega['g']
    solver_u.solve()
    omega.change_scales(ball.dealias)
    omega['g'] = d3.curl(u)['g']
    norm = reducer.global_max(d3.integ(omega@omega)['g'])/ball.volume
    omega['g'] /= np.sqrt(norm)
    omega.change_scales(1)
    data = omega.allgather_data(layout=weight_layout).flatten().reshape((-1, 1))
    random_point.append(data)
    B0.fill_random('g')
    B0.low_pass_filter(scales=0.25);B0['c'];B0['g']
    solver_A.solve()
    if case=='A':
        B0.change_scales(1)
        B0['g'] = A['g']
    elif case=='B0' or case=='B':
        B0.change_scales(ball.dealias)
        B0['g'] = d3.curl(A)['g']
    norm = reducer.global_max(d3.integ(B0@B0)['g'])/ball.volume
    B0['g'] /= np.sqrt(norm)
    B0.change_scales(1)
    data = B0.allgather_data(layout=weight_layout).flatten().reshape((-1, 1))
    random_point.append(data)
    return random_point

###############
# Taylor test #
###############
if test:
    point_0 = random_point()
    point_p = random_point()
    residual = []
    cost_0 = cost(*point_0)
    grad_0 = grad(*point_0)
    dJ = np.vdot(grad_0[0], point_p[0]) + np.vdot(grad_0[1], point_p[1])
    eps = 1e-3
    eps_list = []
    for i in range(10):
        eps_list.append(eps)
        point = [point_0[j] + eps*point_p[j] for j in range(2)]
        cost_p = cost(*point)
        residual.append(np.abs(cost_p - cost_0 - eps*dJ))
        eps /= 2
    regression = linregress(np.log(eps_list), y=np.log(residual))
    logger.info('Result of Taylor test %f' % (regression.slope))

# Perform the optimisation
point = random_point()
verbosity = 2*(comm.rank==0)
log_verbosity = 1*(comm.rank==0)
problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-2, log_verbosity=1)
sol = optimizer.run(problem_opt, initial_point=point)

# Make data directory
data_dir = Path('case_{0:s}_data_Rm_{1:5.03e}'.format(case, Rm))
if rank == 0:  # only do this for the 0th MPI process
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

snapshots = solver.evaluator.add_file_handler(Path(data_dir, 'snapshots'), sim_dt = 0.1)
snapshots.add_task(u, name='u')
snapshots.add_task(omega, name='omega')
snapshots.add_task(B, name='B')
snapshots.add_task(A, name='A')

timeseries = solver.evaluator.add_file_handler(Path(data_dir, 'timeseries'), sim_dt = 1e-3)
timeseries.add_task(d3.integ(B@B)/ball.volume, name='B_int')
timeseries.add_task(d3.integ(A@A)/ball.volume, name='A_int')
cost(*sol.point)
