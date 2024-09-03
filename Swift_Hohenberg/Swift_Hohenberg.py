"""
The script solves the optimal intial condition problem from 
'Connection between nonlinear energy optimization and instantons'
D. Lecoanet and R.R. Kerswell - Phys. Rev. E 97, 012212, 2018
This example finds the optimal condition that maximises the time-integrated energy
"""
import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
import pymanopt
from pymanopt.optimizers import ConjugateGradient
from pymanopt.tools.diagnostics import check_gradient
import uuid
import scipy.sparse as sp

# Just for now
import sys
sys.path.append('../manifolds')
from Generalised_Stiefel import Generalised_Stiefel

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# Parameters
N = 256
dealias = 2
timestep = 0.05
timestepper = d3.SBDF2
test = True

NIter = int(50/timestep)
E0 = 0.2159

coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 12*np.pi), dealias=dealias)

# Global weight matrix
weight = 6*np.ones((N))*np.pi
weight[0] *= 2
weight /= 6
x, = dist.local_grids(xbasis)
u = dist.Field(name='u',bases=xbasis)
cost_t = dist.Field(name='cost_t')
u0 = dist.Field(name='u0',bases=xbasis)
dx = lambda A: d3.Differentiate(A,coords['x'])
a = -0.3

problem = d3.IVP([u,cost_t], namespace=locals())
problem.add_equation("dt(u) + u + 2*dx(dx(u)) + dx(dx(dx(dx(u)))) - a*u = 1.8*u**2 - u**3")
problem.add_equation("dt(cost_t) = d3.integ(0.5*u**2)")
solver = problem.build_solver(timestepper)

# Cost functional
J = -cost_t
Jadj = J.evaluate().copy_adjoint()
checkpoints = {u: []}
timestep_function = lambda : timestep
# Define cost and gradient routines
def forward(vec):
    # Reset solver
    solver.reset()
    solver.stop_iteration = NIter
    # Scale to have energy E0
    u['c'] = vec[:,0]*np.sqrt(E0/0.5)
    # Uncomment to check norms
    # print('sin(0)' ,vec[1])
    # print('Norm', E0*np.vdot(vec,weight_sp@vec))
    # print('Dedalus', (1/12*d3.integ(u**2))['g'])
    cost_t['c'] = 0
    checkpoints[u].clear()
    solver.evolve(timestep_function=timestep_function, checkpoints=checkpoints)
    cost = np.max(J['g'])
    return cost

def backward():
    # Reset adjoint solver
    solver.iteration = NIter
    # Accumulate cotangents
    cotangents = {}
    Jadj['g'] = 1
    cotangents[J] = Jadj
    id = uuid.uuid4()
    _, cotangents =  J.evaluate_vjp(cotangents,id=id,force=True)
    cotangents = solver.compute_sensitivities(cotangents, checkpoints=checkpoints)
    return cotangents[u]['c']*np.sqrt(E0/0.5)

# Manifold for optimisation
weight_sp = sp.diags(weight.flatten())
weight_inv = sp.diags(1/weight.flatten())
manifold = Generalised_Stiefel(N, 1, weight_sp, Binv=weight_inv, k=1, retraction="polar")

@pymanopt.function.numpy(manifold)  
def cost(vecU):
    return forward(vecU)

@pymanopt.function.numpy(manifold)
def grad(vecU):
    # Can comment cost evaluation if sure that optimiser always runs cost before gradient internally.
    # forward(vecU)
    grad = backward().reshape((N,1))
    return grad

problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

# check_gradient(problem)
verbosity = 2
log_verbosity = 1

# Must create initial point manually to avoid sin(0) mode being populated
u.fill_random(layout='c')
norm = np.sqrt(np.max((d3.integ(u**2)/6)['g']))
initial_point = u['c']/norm
initial_point = initial_point.reshape(-1, 1)

costs = []
E_list = [0.2159*0.98,0.2159*1.02]
opt_points = []
for E0 in E_list:
    # Perform the optimisation
    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity, min_gradient_norm=1e-6)
    sol = optimizer.run(problem, initial_point=initial_point)
    opt_points.append(sol.point)
    initial_point = sol.point
    costs.append(sol.cost)

# Get output for optimal seed (E1)
snapshots_E1 = solver.evaluator.add_file_handler('snapshots_E1', sim_dt = 0.5, mode='overwrite')
snapshots_E1.add_task(u)
cost(opt_points[0])

# Get output for optimal seed (E2)
snapshots_E2 = solver.evaluator.add_file_handler('snapshots_E2', sim_dt = 0.5, mode='overwrite')
snapshots_E2.add_task(u)
cost(opt_points[1])