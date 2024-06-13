"""
KdVB inverse problem from 'Iterative Methods for Navier–Stokes Inverse Problems', 
L. O’Connor, D. Lecoanet, E.H. Anders, K.C. Augustson, K.J. Burns, G.M. Vasil, 
J.S. Oishi, B.P. Brown - Phys. Rev. E 109, 045108, 2024

Runs with a shorter final time T=1, rather than T=3pi.
"""
import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import linregress
import uuid

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# Parameters
N = 128
dealias = 1
timestep = 1e-2
timestepper = d3.SBDF2
# T = 3*np.pi
T = 1 # Shorter final time so example runs quickly
NIter = int(T/timestep)
test = True

# Domain and bases
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 4*np.pi), dealias=dealias)
x, = dist.local_grids(xbasis)

# Fields and substitutions
u = dist.Field(name='u',bases=xbasis)
u0 = dist.Field(name='u0',bases=xbasis)
dx = lambda A: d3.Differentiate(A,coords['x'])
a = 0.02
b = 0.04

# Problem
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) - a*dx(dx(u)) + b*dx(dx(dx(u))) = -0.5*dx(u**2)")
solver = problem.build_solver(timestepper)

# Cost functional
J = d3.integ((u-u0)*(u-u0))
Jadj = J.evaluate().copy_adjoint()

# Define cost and gradient routines
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
    u['c'] = vec
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
    return solver.state_adj[0]['c']

# Generate target
u0['g'] = 3/np.cosh((x-np.pi)/(2*np.sqrt(b)))**2
forward(u0['c'])
u0['g'] = u['g']

if test:
    # Taylor test
    vec0 = np.zeros(N)
    vecp = np.random.rand(N)

    cost0 = forward(vec0)
    grad0 = backward()

    eps = 1e-4
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
    plt.savefig('KdVB_Taylor_test.png',dpi=200)

    print('######## Taylor Test Results ########')
    print('First order  : ',linregress(np.log(size), np.log(first)).slope)
    print('Second order : ',linregress(np.log(size), np.log(second)).slope)
    print('#####################################')

# Optimisation
# Define cost_gradient function
def cost_grad(vec):
    cost = forward(vec)
    grad = backward()
    return cost, grad
# Set options
opts = {'disp': True, 'maxiter':200}
# Run the optimisation
sol = optimize.minimize(cost_grad, x0 = np.zeros(N),jac=True,method='L-BFGS-B',tol=1e-8,options=opts)
# Plot the true initial condition, and the reconstructed initial condition obtained via optimisation
u['c'] = sol.x

fig = plt.figure(figsize=(6, 4))
plt.plot(x,3/np.cosh((x-np.pi)/(2*np.sqrt(b)))**2,label=r'True initial condition')
plt.plot(x,u['g'],'x',label=r'Reconstructed initial condition')
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$u(x)$')
plt.savefig('KdVB_reconstructed_initial_condition.png',dpi=200)