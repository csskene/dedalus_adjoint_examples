"""
This script solves the mixing optimisation problem presented in
'Optimal mixing in two-dimensional stratified plane Poiseuille flow at finite Peclet and Richardson numbers'
F. Marcotte and C.P. Caulfield - JFM 853, 359-385, 2018

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
Peinv = 1./100
Ri = 0.05
dealias = 1
T = 0.1
timestep = 1e-3
timestepper = d3.SBDF2
NIter = int(T/timestep)
Niter = 180
E0 = 0.02
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
rho  = dist.Field(name='rho',bases=(xbasis,ybasis))
tau_rho1 = dist.Field(name='tau_rho1',bases=(xbasis))
tau_rho2 = dist.Field(name='tau_rho2',bases=(xbasis))
tau_rho  = dist.Field(name='tau_b',bases=(xbasis))
cost_t  = dist.Field(name='cost_t')

dx = lambda A: d3.Differentiate(A,coords['x'])
dy = lambda A: d3.Differentiate(A,coords['y'])

# Fields for the mix norm
psi = dist.Field(name='psi',bases=(xbasis,ybasis))
tau_psi = dist.Field(name='tau_psi') 
tau_psi1 = dist.Field(name='tau_psi1',bases=(xbasis))
tau_psi2 = dist.Field(name='tau_psi2',bases=(xbasis))

# Lift
lift_basis = ybasis.derivative_basis(1)
lift = lambda A : d3.Lift(A, lift_basis, -1)
lift_basis2 = ybasis.derivative_basis(2)
lift2 = lambda A : d3.Lift(A, lift_basis2, -1)
uy = dy(u) + lift(tau_u1)
vy = dy(v) + lift(tau_v1)
rhoy = dy(rho) + lift(tau_rho1)
psi_y = dy(psi) + lift(tau_psi1)

# Just for now
Reinv_f = dist.Field(name='Reinv_f',bases=(xbasis,ybasis))
Reinv_f['g'] = Reinv 

# Problems
problem = d3.IVP([u,v,rho,p,tau_u1,tau_u2,tau_v1,tau_v2,tau_rho1,tau_rho2,tau_p,cost_t], namespace=locals())
problem.add_equation("dt(u) - Reinv*(dx(dx(u))+dy(uy)) + dx(p) + lift2(tau_u2) = Reinv_f*2 -u*dx(u) - v*dy(u)")
problem.add_equation("dt(v) - Reinv*(dx(dx(v))+dy(vy)) + dy(p) + lift2(tau_v2) + Ri*rho = -u*dx(v) - v*dy(v)")
problem.add_equation("dt(rho) - Peinv*(dx(dx(rho))+dy(rhoy)) + lift2(tau_rho2) = -u*dx(rho) - v*dy(rho)")
problem.add_equation("dt(cost_t) = integ(u**2+v**2)")
problem.add_equation("dx(u) + vy + tau_p = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("u(y=-1) = 0")
problem.add_equation("u(y=1)  = 0")
problem.add_equation("v(y=-1) = 0")
problem.add_equation("v(y=1)  = 0")
problem.add_equation("rhoy(y=-1) = 0")
problem.add_equation("rhoy(y=1)  = 0")
solver = problem.build_solver(timestepper)
solver.stop_iteration = NIter

problem_mix_norm = d3.LBVP([psi, tau_psi1, tau_psi2, tau_psi], namespace=locals())
problem_mix_norm.add_equation("dx(dx(psi)) + dy(psi_y) + lift(tau_psi2) + tau_psi = rho")
problem_mix_norm.add_equation("dy(psi)(y=-1) = 0")
problem_mix_norm.add_equation("dy(psi)(y=1) = 0")
problem_mix_norm.add_equation("integ(psi) = 0")
solve_psi = problem_mix_norm.build_solver()
solve_psi.solve()

# Base-flow
u0['g'] = 1-y**2

# Cost functional
alpha = 0.5
J = alpha/2*d3.integ(dx(psi)**2+dy(psi)**2) + (1-alpha)/(2*T)*cost_t
Jadj = J.evaluate().copy_adjoint()
timestep_function = lambda : timestep
checkpoints = {u: [], v: [], rho: []}

# cost and gradient routines
def forward(vec):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(np.squeeze(vec), 2)
    u['c'] = u0['c'] + vec_split[0].reshape((Nx,Ny))
    v['c'] = vec_split[1].reshape((Nx,Ny))
    rho['g'] = -0.5*erf(y/0.125)
    cost_t['g'] = 0
    # Evolve IVP
    checkpoints[u].clear()
    checkpoints[v].clear()
    checkpoints[rho].clear()
    solver.evolve(timestep_function=timestep_function,checkpoints=checkpoints)
    solve_psi.solve()
    cost = np.max(J['g'])
    return cost

def backward():
    # Final time condition
    Jadj['g'] = 1
    cotangents = {}
    cotangents[J] = Jadj
    id = uuid.uuid4()
    _, cotangents =  J.evaluate_vjp(cotangents,id=id,force=True)
    cotangents = solve_psi.compute_sensitivities(cotangents)
    cotangents = solver.compute_sensitivities(cotangents,checkpoints=checkpoints)
    return np.hstack([cotangents[state]['c'].flatten() for state in [u,v]])

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
    print(np.array(first)/np.array(size),np.vdot(grad0,vecp))
    fig = plt.figure(figsize=(6, 4))
    plt.loglog(size,first,label=r'First order')
    plt.loglog(size,second,label=r'Second order')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'Error')
    plt.title(r'Taylor test')
    plt.legend()
    plt.show()
    # plt.savefig('optimal_mixing_Taylor_test.png',dpi=200)

    print('######## Taylor Test Results ########')
    print('First order  : ',linregress(np.log(size), np.log(first)).slope)
    print('Second order : ',linregress(np.log(size), np.log(second)).slope)
    print('#####################################')