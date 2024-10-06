"""
Usage:
    optimal_dynamo.py [options]

Options:
    --Rm=<Rm>                   Magnetic Reynolds number [default: 65]
"""
import dedalus.public as d3
import numpy as np
import uuid
from mpi4py import MPI
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
import scipy.sparse as sp
dtype = np.float64
import matplotlib.pyplot as plt
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.stiefel import Stiefel
from pymanopt.manifolds.product import Product
from pymanopt.tools.diagnostics import check_gradient
import logging
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
import os
from pathlib import Path

from docopt import docopt
args = docopt(__doc__)

# Just for now
import sys
sys.path.append('../manifolds')
from generalized_stiefel import GeneralizedStiefel

from dedalus.extras.flow_tools import GlobalArrayReducer
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

lmax = 15
Nphi = 2*(lmax+1)
Ntheta = lmax+1
Nr = 16
Ro = 1
dealias = 1
timestep = 1e-3

Rm = float(args['--Rm'])
if rank==0:
    print('Running with Rm: %f' % Rm)

factors = [[ncpu//i,i] for i in range(1,int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
ball = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=Ro, dealias=dealias, dtype=dtype)
sphere = ball.surface
phii, theta, r = dist.local_grids(ball)

# Weights
weight_theta = ball.global_colatitude_weights(dist)
weight_r = ball.global_radial_weights(dist)
weight = Ro**3*weight_theta*weight_r*np.ones((Nphi, 1, 1))*(2*np.pi/Nphi)
vol_test = np.sum(weight)
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = ball.volume

er = dist.VectorField(coords, name='er')
er['g'][2] = 1

omega  = dist.VectorField(coords, name='omega', bases=ball)
omega_clean  = dist.VectorField(coords, name='omega_clean', bases=ball)

u = dist.VectorField(coords, name='u', bases=ball)
tau_u = dist.VectorField(coords, name='tau_u', bases=sphere)
V = dist.Field(name='V', bases=ball)
tau_V = dist.Field(name='tau_V')
Vs = dist.Field(name='Vs', bases=sphere)

A       = dist.VectorField(coords, name='A', bases=ball)
tau_A   = dist.VectorField(coords, name='tau_A', bases=sphere)
phi     = dist.Field(name='phi', bases=ball)

tau_phi2     = dist.Field(name='tau_phi2', bases=sphere)

tau_phi = dist.Field(name='tau_phi')
B = d3.curl(A)

ell_func = lambda ell: ell+1
ellmult = lambda A: d3.SphericalEllProduct(A, coords, ell_func) # mult by (ell+1)

lift = lambda A : d3.Lift(A, ball, -1)

problem_omega = d3.LBVP([omega_clean, phi, tau_phi, tau_phi2], namespace=locals())
problem_omega.add_equation("omega_clean + grad(phi) = omega")
problem_omega.add_equation("div(omega_clean) + tau_phi + lift(tau_phi2) = 0")
problem_omega.add_equation("integ(phi) = 0")
problem_omega.add_equation("phi(r=Ro) = 0")
solver_omega = problem_omega.build_solver()

problem_u = d3.LBVP([u, tau_u, V, tau_V], namespace=locals())
problem_u.add_equation("-lap(u) + grad(V) + lift(tau_u) = curl(omega_clean)")
problem_u.add_equation("div(u) + tau_V = 0")
problem_u.add_equation("integ(V) = 0")
problem_u.add_equation("u(r=Ro) = 0")
solver_u = problem_u.build_solver()

problem = d3.IVP([A, u, phi, tau_phi, tau_A], namespace=locals())
problem.add_equation("dt(A) - lap(A) + grad(phi) + lift(tau_A) = Rm*cross(u, curl(A))")
problem.add_equation("dt(u) = 0")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")
problem.add_equation("radial(grad(A)(r=Ro)) + ellmult(A)(r=Ro)/Ro = 0")
solver = problem.build_solver(d3.SBDF2)

J = -np.log(d3.integ(A@A))
Jadj = J.evaluate().copy_adjoint()

# For parallelisation
layout = dist.layouts[-1]

local_slice = layout.slices(u.domain, scales=1)
gshape      = layout.global_shape(u.domain, scales=1)
lshape      = layout.local_shape(u.domain, scales=1)

timestep_function = lambda : timestep
checkpoints = {A: []}
def forward(vecs):
    vec_split = np.split(np.squeeze(vecs[0]), 3)
    for i in range(3):
        omega[layout][i]  = vec_split[i].reshape(lshape)
    vec_split = np.split(np.squeeze(vecs[1]), 3)
    for i in range(3):
        A[layout][i] = vec_split[i].reshape(lshape)
    solver_omega.solve()
    solver_u.solve()
    # print('max', np.max(V['g']))
    # print('curl u', np.max((d3.curl(u)-omega)['g']))
    # print('curl u2', np.max((d3.curl(u)-omega_clean)['g']))
    # print('div A', np.max(d3.div(A)['g']))
    # Reset solver
    solver.reset()
    for key in list(checkpoints.keys()):
        checkpoints[key].clear()
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
    cotangents = solver_omega.compute_sensitivities(cotangents)
    grad_omega = np.hstack([(cotangents[omega][layout][idx]).flatten() for idx in [0,1,2]])
    grad_B = np.hstack([(cotangents[A][layout][idx]).flatten() for idx in [0,1,2]])
    return [grad_omega, grad_B]

N = 3*np.prod(gshape)
# Create the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))

manifold_GS = GeneralizedStiefel(N, 1,weight_sp, Binv=weight_inv, retraction="polar")
manifold_GS2 = GeneralizedStiefel(N, 1,weight_sp/ball.volume, Binv=weight_inv*ball.volume, retraction="polar")
manifold = Product([manifold_GS2, manifold_GS])

# TODO: Tidy up parallelisation with scatters/gathers?
@pymanopt.function.numpy(manifold)  
def cost(vec_omega, vec_B):
    local_vec = np.split(vec_omega, 3)
    local_vec_U = np.hstack([(local_vec[idx].reshape(gshape)[local_slice]).flatten() for idx in [0,1,2]])
    local_vec = np.split(vec_B, 3)
    local_vec_B = np.hstack([(local_vec[idx].reshape(gshape)[local_slice]).flatten() for idx in [0,1,2]])
    return forward([local_vec_U, local_vec_B])

omega_grad = omega.copy_adjoint()
B_grad = A.copy_adjoint()

@pymanopt.function.numpy(manifold)
def grad(vecU, vecB):
    # Can comment cost evaluation if sure that optimiser always runs cost before gradient internally.
    # cost(vec)
    local_grad = backward()
    local_vec = np.split(local_grad[0], 3)
    for i in range(3):
        omega_grad[layout][i] = local_vec[i].reshape(lshape)
    grad_omega = (omega_grad.allgather_data(layout=layout)).flatten()
    # Reshape to manifold shape
    grad_omega = grad_omega.reshape((-1, 1))

    local_vec = np.split(local_grad[1], 3)
    for i in range(3):
        B_grad[layout][i] = local_vec[i].reshape(lshape)
    grad_B = (B_grad.allgather_data(layout=layout)).flatten()
    # Reshape to manifold shape
    grad_B = grad_B.reshape((-1, 1))
    return [grad_omega, grad_B]

# Parallel-safe random point and tangent-vector
random_point = manifold.random_point()
random_point = comm.bcast(random_point, root=0)
random_tangent_vector = manifold.random_tangent_vector(random_point)
random_tangent_vector = comm.bcast(random_tangent_vector, root=0)

verbosity = 2*(comm.rank==0)
log_verbosity = 1*(comm.rank==0)
problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=400, min_gradient_norm=1e-3, log_verbosity=1)
solver.stop_iteration = int(2/timestep)
# check_gradient(problem_opt, x=random_point, d=random_tangent_vector)

A.fill_random(layout='g')
A.low_pass_filter(scales=0.5)
omega.fill_random(layout='g')
omega.low_pass_filter(scales=0.5)
omega['g'] = d3.curl(omega)['g']
vec1 = omega.allgather_data(layout=layout).flatten().reshape(-1, 1)
vec2 = A.allgather_data(layout=layout).flatten().reshape(-1, 1)
vec1 /= np.sqrt(vec1.T@weight_sp@vec1)
vec2 /= np.sqrt(vec2.T@weight_sp@vec2)
initial_point=[vec1, vec2]
sol = optimizer.run(problem_opt, initial_point=initial_point)

# Get outputs

# Make data directory
data_dir = 'data_Rm_{0:5.02e}'.format(Rm)
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

snapshots = solver.evaluator.add_file_handler(Path("{0:s}/snapshots".format(data_dir)), sim_dt = 0.1)
snapshots.add_task(u, name='u')
snapshots.add_task(omega, name='omega')
snapshots.add_task(B, name='B')

timeseries = solver.evaluator.add_file_handler(Path("{0:s}/timeseries".format(data_dir)), sim_dt = 1e-3)
timeseries.add_task(d3.integ(A@A), name='A_int')
timeseries.add_task(d3.integ(B@B), name='B_int')
cost(*sol.point)
