"""
The script solves the optimal intial condition problem from 
'Connection between nonlinear energy optimization and instantons'
D. Lecoanet and R.R. Kerswell - Phys. Rev. E 97, 012212, 2018
This example finds the optimal initial condition that maximises the 
time-integrated energy

Usage:
    Swift_Hohenberg.py [options]

Options:
    --test                      Run test only
"""
import logging
import numpy as np
from dedalus import public as d3
import scipy.sparse as sp
from scipy.optimize import minimize
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from checkpoint_schedules import SingleMemoryStorageSchedule
from docopt import docopt
from dedalus.tools import adjoint as d3_adj
logger = logging.getLogger(__name__)

# Parameters
N = 256
dealias = 2
timestep = 0.05
timestepper = d3.SBDF2
total_steps = int(50/timestep)
E0 = 0.2159*1.01
a = -0.3

# Parse arguments
args = docopt(__doc__)
test = args['--test']

# Domain and bases
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 12*np.pi), dealias=dealias)
x, = dist.local_grids(xbasis)

# Global weight matrix in coefficient space
weight = 6*np.ones((N))*np.pi
weight[0] *= 2
weight *= 0.5/6/E0

# Fields and substitutions
u = dist.Field(name='u', bases=xbasis)
cost_t = dist.Field(name='cost_t')
u0 = dist.Field(name='u0', bases=xbasis)
dx = lambda A: d3.Differentiate(A, coords['x'])

# Problem and solver
problem = d3.IVP([u, cost_t], namespace=locals())
problem.add_equation("dt(u) + u + 2*dx(dx(u)) + dx(dx(dx(dx(u)))) - a*u = 1.8*u**2 - u**3")
problem.add_equation("dt(cost_t) = 0.5*integ(u**2)")
solver = problem.build_solver(timestepper)

# Cost functional
J = -cost_t

# Set up direct adjoint loop
dal = d3_adj.direct_adjoint_loop(solver, total_steps, timestep, J, adjoint_dependencies=[u])

# Set up checkpointing
create_schedule = lambda : SingleMemoryStorageSchedule()
manager = d3_adj.CheckpointingManager(create_schedule, dal)  # Create the checkpointing manager.

# Set up manifold
weight_sp = sp.diags(weight.flatten())
weight_inv = sp.diags(1/weight.flatten())
GeneralizedStiefel = d3_adj.GeneralizedStiefelManifold()
manifold = GeneralizedStiefel(N, 1, weight_sp, Binv=weight_inv, retraction="qr")

num_fun_evals = 0
num_grad_evals = 0
# Set up cost and gradient routines
@pymanopt.function.numpy(manifold)
def cost(vec_u):
    global num_fun_evals
    num_fun_evals += 1
    u['c'] = vec_u[:, 0]
    cost_t['c'] = 0
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
    return cotangents[u]['c'].reshape((-1, 1))

def random_point(seed=None):
    # Random point for u
    # Must use to ensure sin(0) mode has zero coefficient
    u.fill_random(layout='g', seed=seed)
    u.low_pass_filter(scales=0.6)
    data = u['c'].copy()
    data /= np.sqrt(np.vdot(data, weight_sp@data))
    return data.reshape((-1, 1))

if test:
    point0 = random_point(42)
    pointp = random_point(43)
    slope, eps_list, residual = d3_adj.Taylor_test(cost, grad, point0, pointp)
    logger.info('Result of Taylor test %f' % (slope))
    np.savez('swift_test', eps=np.array(eps_list), residual=np.array(residual))
else:
    # Perform the optimisation
    initial_point = random_point()
    problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
    optimizer = ConjugateGradient(verbosity=2, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-3, log_verbosity=1)
    sol = optimizer.run(problem_opt, initial_point=initial_point)
    logger.info('Number of function evaluations {0:d}'.format(num_fun_evals))
    logger.info('Number of gradient evaluations {0:d}'.format(num_grad_evals))

    # Fix the phase so the minimum value is in the centre of the domain
    u['c'] = sol.point[:, 0]
    u.change_scales(1)
    index = np.argmin(u['g'])
    x_guess = x[index]
    # Find x where u is minimised
    eval_point = lambda A: np.max((u(x=A[0])).evaluate()['g'])
    res = minimize(eval_point, x0=x_guess)
    # Adjust coefficients to shift the solution
    sin_cos_factors = u['c'].reshape(-1, 2)
    # Shift on the native grid
    shift = (res.x - 6*np.pi)/6
    native_wavenumbers = xbasis.native_wavenumbers[::2]
    shift_freqs = shift*native_wavenumbers
    cos_vals = np.cos(shift_freqs)
    sin_vals = np.sin(shift_freqs)
    shift_mats = np.array([[cos_vals, -sin_vals], [-sin_vals, -cos_vals]])
    # Apply shift matrices to sin_cos_factors
    sin_cos_factors_shifted = np.einsum('ijn,nj->ni', shift_mats, sin_cos_factors)
    sol.point[:, 0] = sin_cos_factors_shifted.reshape(-1)
    
    # Get outputs
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt = 0.5, mode='overwrite')
    snapshots.add_task(u)
    cost(sol.point)
