"""
This script finds transient growth that maximises the p_norm of the energy for 
plane Poiseuille channel flow following
'Localization of flow structures using ∞-norm optimization'
D.P.G. Foures, C.P. Caulfield, and P.J. Schmid - JFM 729, 672–701, 2013

For now just sets up the problem and checks that the gradient passes the Taylor test
"""
import numpy as np
from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
from scipy.stats import linregress
# import pymanopt
# from pymanopt.manifolds import Sphere
# from pymanopt.optimizers import ConjugateGradient
from scipy.special import erf
import uuid

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
# Parameters
Nx = 128
Ny = 128
Reinv = 1./100
dealias = 1
T = 0.1
timestep = 1e-3
timestepper = d3.SBDF2
NIter = int(T/timestep)
Niter = 180
p_norm = 30
test = True

# Domain and bases
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, 2*np.pi), dealias=dealias)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-1, 1), dealias=dealias)
x, y = dist.local_grids(xbasis, ybasis)

# Global weight matrix
weight = 6*np.ones((Nx))*np.pi
weight[0] *= 2

# Fields and substitutions
u  = dist.Field(name='u',bases=(xbasis,ybasis))
v  = dist.Field(name='v',bases=(xbasis,ybasis))
tau_u1 = dist.Field(name='tau_u1',bases=(xbasis))
tau_u2 = dist.Field(name='tau_u2',bases=(xbasis))
tau_v1 = dist.Field(name='tau_v1',bases=(xbasis))
tau_v2 = dist.Field(name='tau_v2',bases=(xbasis))
p  = dist.Field(name='p',bases=(xbasis,ybasis))
tau_p  = dist.Field(name='tau_p')
u0 = dist.Field(name='u0',bases=(xbasis,ybasis))

dx = lambda A: d3.Differentiate(A,coords['x'])
dy = lambda A: d3.Differentiate(A,coords['y'])

# Lift
lift_basis = ybasis.derivative_basis(1)
lift = lambda A : d3.Lift(A, lift_basis, -1)
lift_basis2 = ybasis.derivative_basis(2)
lift2 = lambda A : d3.Lift(A, lift_basis2, -1)
uy = dy(u) + lift(tau_u1)
vy = dy(v) + lift(tau_v1)

# Problems
problem = d3.IVP([u,v,p,tau_u1,tau_u2,tau_v1,tau_v2,tau_p], namespace=locals())
problem.add_equation("dt(u) - Reinv*(dx(dx(u))+dy(uy)) + dx(p) + lift2(tau_u2) = -u0*dx(u) - u*dx(u0) - v*dy(u0)")
problem.add_equation("dt(v) - Reinv*(dx(dx(v))+dy(vy)) + dy(p) + lift2(tau_v2) = -u0*dx(v)")
problem.add_equation("dx(u) + vy + tau_p = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("u(y=-1) = 0")
problem.add_equation("u(y=1)  = 0")
problem.add_equation("v(y=-1) = 0")
problem.add_equation("v(y=1)  = 0")
solver = problem.build_solver(timestepper)
solver.stop_iteration = NIter

# Base-flow
u0['g'] = 1-y**2

# Cost functional
J = d3.integ((u**2 + v**2)**p_norm)**(1/p_norm)
Jadj = J.evaluate().copy_adjoint()
checkpoints = {u: [], v: []}
timestep_function = lambda : timestep
# cost and gradient routines
states_u   = []
states_v   = []
def forward(vec):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(np.squeeze(vec), 2)
    u['c'] = vec_split[0].reshape((Nx,Ny))
    v['c'] = vec_split[1].reshape((Nx,Ny))
    # Evolve
    checkpoints[u].clear()
    checkpoints[v].clear()
    solver.evolve(timestep_function=timestep_function,checkpoints=checkpoints)
    cost = np.max(J['g'])
    return cost

def backward():    
    # Final time condition
    Jadj['g'] = 1
    cotangents = {}
    cotangents[J] = Jadj
    id = uuid.uuid4()
    _, cotangents =  J.evaluate_vjp(cotangents,id=id,force=True)
    cotangents = solver.compute_sensitivities(cotangents, checkpoints=checkpoints)
    return np.hstack([cotangents[f]['c'].flatten() for f in [u,v]])

if test:
    # Taylor test
    u.fill_random('g');u['c'];u['g']
    vecp = np.hstack((u['c'].flatten(),v['c'].flatten()))
    u.fill_random('g');u['c'];u['g']
    vec0 = np.hstack((u['c'].flatten(),v['c'].flatten()))

    cost0 = forward(vec0)
    grad0 = backward()

    eps = 1e-3
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
    plt.savefig('transient_growth_Lp_Taylor_test.png',dpi=200)

    print('######## Taylor Test Results ########')
    print('First order  : ',linregress(np.log(size), np.log(first)).slope)
    print('Second order : ',linregress(np.log(size), np.log(second)).slope)
    print('#####################################')