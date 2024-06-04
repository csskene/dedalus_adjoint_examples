import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt

import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.product import Product

from pymanopt.tools.diagnostics import check_gradient

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# Parameters
N = 256
dealias = 1
timestep = 0.05
timestepper = d3.SBDF2

NIter = 50/timestep
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

states = []

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
    cost = -np.max(cost_t['c'])
    return cost

def backward():
    # Reset adjoint solver
    solver.iteration = NIter
    solver.timestepper._LHS_params = False
    # Adjoint final time condition
    solver.state_adj[0]['c'] = 0
    solver.state_adj[1]['c'] = -1
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

def inner_product(x,y):
    return np.dot(x,y)

vec0 = np.random.rand(N)
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
print(np.vdot(vec0,weight*vec0))
vec0 *= np.sqrt(weight)
costs = []
E_list = [0.2159*0.98,0.2159*1.02]
vecs = []
for E0 in E_list:
    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity, min_gradient_norm=1e-3)
    sol = optimizer.run(problem, initial_point = vec0)
    costs.append(sol.cost)
    vecs.append(sol.point)

u0['c'] = sol.point/np.sqrt(weight) 
plt.plot(u0['g'])
plt.show()
for (vec0,E0) in zip(vecs,E_list):
    # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    solver.sim_time = 0
    # problem.time['g']=0
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False
    solver.stop_iteration = NIter*4
    u['c'] = vec0/np.sqrt(weight)*np.sqrt(E0/0.5)
    cost_t['c'] = 0
    states.clear()
    KE = 0.5*d3.integ(u*u)/6
    KE_list = [np.max(KE['g'])]
    t_list = [0]
    try:
        while solver.proceed:
            solver.step(timestep)
            KE_list.append(np.max(KE['g']))
            t_list.append(solver.sim_time)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    plt.plot(t_list,KE_list)
plt.show()

# Taylor test
vec0 = np.random.rand(N)
vecp = np.random.rand(N)
u0.fill_random('g');u0['c'];u0['g']
u0.low_pass_filter(0.3)
vec0 = u0['c'] 
vecp = u0['c']
# vec0 = 1e-12*np.sin(x)
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

print(np.vdot(grad0,vecp),np.array(first)/np.array(size))
plt.loglog(size,first,label=r'First order')
plt.loglog(size,second,label=r'Second order')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$Taylor test$')
plt.legend()
plt.show()

from scipy.stats import linregress
print('######## Taylor Test Results ########')
print('First order  : ',linregress(np.log(size), np.log(first)).slope)
print('Second order : ',linregress(np.log(size), np.log(second)).slope)
print('#####################################')