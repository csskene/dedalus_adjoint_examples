"""
Usage:
    kinematic_dynamo.py [options]

Options:
    --Rm=<Rm>                   Magnetic Reynolds number [default: 1]
"""
import numpy as np
from dedalus import public as d3
import logging
import sys
from dedalus.extras.flow_tools import GlobalArrayReducer
import uuid
import scipy.sparse as sp
from pathlib import Path
from mpi4py import MPI
import os
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
reducer = GlobalArrayReducer(MPI.COMM_WORLD)
logger = logging.getLogger(__name__)

import pymanopt
from pymanopt.optimizers import ConjugateGradient

from pymanopt.manifolds.product import Product
from pymanopt.tools.diagnostics import check_gradient
logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Just for now
import sys
sys.path.append('../manifolds')
from generalized_stiefel import GeneralizedStiefel

from docopt import docopt
args = docopt(__doc__)

# Parameters
N = 24
Rm = float(args['--Rm'])
dealias = 3/2
timestep = 5e-3
timestepper = d3.RK222
NIter = int(2/timestep)

if rank==0:
    print('Running with Rm: %f' % Rm)

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

# Global weight matrix (definitely better ways!)
weight = np.ones((N,N,N))*np.pi**3
# Adjust for zeroth rows
for i in range(N):
    for j in range(N):
        for k in range(N):
            p = np.count_nonzero(np.array([i, j, k])==0)
            weight[i, j, k] *= 2**p
# weight /= 1
weight_layout = dist.coeff_layout
local_slice = weight_layout.slices(u.domain, scales=1)
gshape = weight_layout.global_shape(u.domain, scales=1)

problem = d3.IVP([A, phi, tau_phi, u, C], namespace=locals())
problem.add_equation("dt(A) + grad(phi) - lap(A) + C = Rm*cross(u, B)")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(A) = 0")
problem.add_equation("integ(phi) = 0")
problem.add_equation("dt(u) = 0")
solver = problem.build_solver(timestepper)
solver.stop_iteration = NIter

# Problems for finding u from omega
problem_u = d3.LBVP([u, Pi, tau_Pi, C], namespace=locals())
problem_u.add_equation("-lap(u) + grad(Pi) + C = curl(omega)")
problem_u.add_equation("div(u) + tau_Pi = 0")
problem_u.add_equation("integ(Pi) = 0")
problem_u.add_equation("integ(u) = 0")
solver_u = problem_u.build_solver()

J = -np.log(d3.Average(A@A))
Jadj = J.evaluate().copy_adjoint()
timestep_function = lambda : timestep
checkpoints = {A: []}
def forward(vecs):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(vecs[0], 3)
    for i in range(3):
        omega['c'][i] = vec_split[i].reshape(omega[weight_layout].shape[1:])
    vec_split = np.split(vecs[1], 3)
    for i in range(3):
        A['c'][i] = vec_split[i].reshape(A[weight_layout].shape[1:])
    for key in list(checkpoints.keys()):
        checkpoints[key].clear()
    # Remove divergence and average
    solver_u.solve()
    # Evolve IVP and compute cost
    solver.evolve(timestep_function=timestep_function, checkpoints=checkpoints)
    cost = reducer.global_max(J['g'])
    return cost

def backward():
    Jadj['g'] = 1
    cotangents = {}
    cotangents[J] = Jadj
    _, cotangents =  J.evaluate_vjp(cotangents, id=uuid.uuid4(), force=True)
    cotangents = solver.compute_sensitivities(cotangents, checkpoints=checkpoints)
    cotangents = solver_u.compute_sensitivities(cotangents)
    grad_omega = np.hstack([(cotangents[omega]['c'][idx]).flatten() for idx in range(3)])
    grad_B     = np.hstack([(cotangents[A]['c'][idx]).flatten() for idx in range(3)])
    return [grad_omega, grad_B]

# Create the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))
manifold_GS = GeneralizedStiefel(3*N**3, 1, weight_sp,Binv=weight_inv, retraction="polar")
manifold = Product([manifold_GS, manifold_GS])

# TODO: Tidy up parallelisation with scatters/gathers?
@pymanopt.function.numpy(manifold)  
def cost(vecU, vecB):
    vec = [vecU, vecB]
    vecu, vecv, vecw = np.split(vec[0], 3)
    vecBx, vecBy, vecBz = np.split(vec[1], 3)
    local_vec_U = np.hstack([(v.reshape(gshape)[local_slice]).flatten() for v in [vecu, vecv, vecw]])
    local_vec_B = np.hstack([(v.reshape(gshape)[local_slice]).flatten() for v in [vecBx, vecBy, vecBz]])
    return forward([local_vec_U, local_vec_B])

u_grad = u.copy_adjoint()
A_grad = A.copy_adjoint()

@pymanopt.function.numpy(manifold)
def grad(vecU, vecB):
    # Can comment cost evaluation if sure that optimiser always runs cost before gradient internally.
    # cost(vec)
    local_grad = backward()
    local_u = np.split(local_grad[0], 3)
    for i in range(3):
        u_grad[weight_layout][i] = local_u[i].reshape(u_grad[weight_layout].shape[1:])
    gradU = u_grad.allgather_data(layout=weight_layout).flatten()
    # Reshape to manifold shape
    gradU = gradU.reshape((-1, 1))

    local_A = np.split(local_grad[1], 3)
    for i in range(3):
        A_grad[weight_layout][i] = local_A[i].reshape(A_grad[weight_layout].shape[1:])
    gradB = A_grad.allgather_data(layout=weight_layout).flatten()
    # Reshape to manifold shape
    gradB = gradB.reshape((-1, 1))
    return [gradU, gradB]

# Parallel-safe random point and tangent-vector
random_point = manifold.random_point()
random_point = comm.bcast(random_point,root=0)
random_tangent_vector = manifold.random_tangent_vector(random_point)
random_tangent_vector = comm.bcast(random_tangent_vector,root=0)

verbosity = 2*(comm.rank==0)
log_verbosity = 1*(comm.rank==0)
problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

# check_gradient(problem_opt, x=random_point, d=random_tangent_vector)
# Random divergence-free, zero average initial condition
for field in [omega, A]:
    field.fill_random(layout='c')
    field.low_pass_filter(scales=0.25)

solver_u.solve()
initial_u = u.allgather_data(layout=weight_layout).flatten() 
initial_A = A.allgather_data(layout=weight_layout).flatten()

# Normalise
initial_u /= np.sqrt(initial_u.T@weight_sp@initial_u)
initial_A /= np.sqrt(initial_A.T@weight_sp@initial_A)
initial_u = initial_u.reshape((-1, 1))
initial_A = initial_A.reshape((-1, 1))
initial_point = [initial_u, initial_A]
optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100, min_gradient_norm=1e-3, log_verbosity=log_verbosity)
sol = optimizer.run(problem_opt, initial_point=initial_point)

if comm.rank==0:
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]
        np.savez('convergence', iterations=iterations, costs=costs, gradient_norms=gradient_norms)

# Get output for optimal seed

# Make data directory
data_dir = 'data_Rm_{0:5.02e}'.format(Rm)
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

snapshots = solver.evaluator.add_file_handler(Path("{0:s}/snapshots".format(data_dir)), sim_dt = 0.1, mode='overwrite')
snapshots.add_task(u, name='u')
snapshots.add_task(B, name='B')
snapshots.add_task(u@d3.curl(u), name='helicity')

timeseries = solver.evaluator.add_file_handler(Path("{0:s}/timeseries".format(data_dir)), sim_dt = 1e-3)
timeseries.add_task(d3.Average(A@A), name='A_int')
timeseries.add_task(d3.Average(B@B), name='B_int')

cost(sol.point[0], sol.point[1])
