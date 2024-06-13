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
from scipy.stats import linregress
import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
import uuid

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# Parameters
N = 256
dealias = 1
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
states = []
# Define cost and gradient routines
def forward(vec):
    # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False
    solver.stop_iteration = NIter
    u['c'] = vec/np.sqrt(weight)*np.sqrt(E0/0.5)
    cost_t['c'] = 0
    states.clear()
    try:
        while solver.proceed:
            states.append(u['c'].copy())
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    cost = np.max(J['g'])
    return cost

def backward():
    # Reset adjoint solver
    solver.iteration = NIter
    solver.timestepper._LHS_params = False
    # Adjoint final time condition
    cotangents = {}
    Jadj['g'] = 1
    cotangents[J] = Jadj
    id = uuid.uuid4()
    _, cotangents =  J.evaluate_vjp(cotangents,id=id,force=True)
    # Zero adjoint state
    for f in solver.state_adj:
        f['c'] *= 0
    # Add contributions from cotangents
    for (i,f) in enumerate(solver.state):
        if f in list(cotangents.keys()):
            solver.state_adj[i]['c'] += cotangents[f]['c'] 
    count = -1
    try:
        while solver.iteration>0:
            u['c'] = states[count]
            count -= 1
            solver.step_adjoint(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    return solver.state_adj[0]['c']/np.sqrt(weight)*np.sqrt(E0/0.5)

if test:
    # Taylor test
    vec0 = np.random.rand(N)
    vecp = np.random.rand(N)
    u0.fill_random('g');u0['c'];u0['g']
    u0.low_pass_filter(0.3)
    vec0 = u0['c']
    vecp = u0['c']
    cost0 = forward(vec0)
    grad0 = backward()

    eps = 1e-1
    costs = []
    size = []
    for i in range(10):
        costp = forward(vec0+eps*vecp)
        costs.append(costp)
        size.append(eps)
        eps /= 2

    first = np.abs(np.array(costs)-cost0)
    second = np.abs(np.array(costs)-cost0 - np.array(size)*np.vdot(grad0,vecp))
    fig = plt.figure(figsize=(6, 4))
    plt.loglog(size,first,label=r'First order')
    plt.loglog(size,second,label=r'Second order')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'Error')
    plt.title(r'Taylor test')
    plt.legend()
    plt.savefig('Swift_Hohenberg_Taylor_test.png',dpi=200)

    print('######## Taylor Test Results ########')
    print('First order  : ',linregress(np.log(size), np.log(first)).slope)
    print('Second order : ',linregress(np.log(size), np.log(second)).slope)
    print('#####################################')

# Perform the optimisation
# Initial guess
vec0 = np.random.rand(N)
# Manifold for optimisation
manifold = Sphere(N)

@pymanopt.function.numpy(manifold)  
def cost(vecU):
    return forward(vecU)

@pymanopt.function.numpy(manifold)
def grad(vecU):
    forward(vecU)
    grad = backward()
    return grad

problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

verbosity = 2
log_verbosity = 1

norm = np.vdot(vec0,weight*vec0)
vec0 /= np.sqrt(norm)
vec0 *= np.sqrt(weight)
costs = []
E_list = [0.2159*1.02]
for E0 in E_list:
    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity, min_gradient_norm=1e-3)
    sol = optimizer.run(problem, initial_point = vec0)
    costs.append(sol.cost)

u0['c'] = sol.point/np.sqrt(weight)
fig = plt.figure(figsize=(6, 4))
plt.plot(x,u0['g'])
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x)$')
plt.title(r'Optimal initial condition')
plt.savefig('Swift_Hohenberg_optimal_initial_condition.png',dpi=200)