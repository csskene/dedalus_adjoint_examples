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
from scipy.special import erf
from mpi4py import MPI
from dedalus.extras.flow_tools import GlobalArrayReducer
import uuid
from dedalus.tools import jacobi

import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.product import Product

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
comm = MPI.COMM_WORLD
reducer = GlobalArrayReducer(comm)

# Parameters
Nx = 128
Ny = 128
Reinv = 1./500
Peinv = 1./500
Ri = 0.05
dealias = 1
T = 5
timestep = 5e-3
timestepper = d3.SBDF2
NIter = int(T/timestep)
E0 = 0.02
test = False

# Domain and bases
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, 2*np.pi), dealias=dealias)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-1, 1), dealias=dealias)
x, y = dist.local_grids(xbasis, ybasis)

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

# Global weight matrix
# Fourier part
weight = np.ones((Nx,Ny))*np.pi
# Adjust for zeroth rows
weight[0,:] *= 2
# Jacobi part
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=(ybasis), adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
weight *= W_field.allgather_data(layout='g').reshape((1,Ny))

weight_layout = dist.layouts[1]
local_slice = dist.coeff_layout.slices(u.domain,scales=1)
gshape = dist.coeff_layout.global_shape(u.domain,scales=1)
local_weight = weight[local_slice]
local_M_matrix = np.sqrt(local_weight)

# Lift
lift_basis = ybasis.derivative_basis(1)
lift = lambda A : d3.Lift(A, lift_basis, -1)
lift_basis2 = ybasis.derivative_basis(2)
lift2 = lambda A : d3.Lift(A, lift_basis2, -1)
uy = dy(u) + lift(tau_u1)
vy = dy(v) + lift(tau_v1)
rhoy  = dy(rho) + lift(tau_rho1)
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
J = -(alpha/2*d3.integ(dx(psi)**2+dy(psi)**2) + (1-alpha)/(2*T)*cost_t)
Jadj = J.evaluate().copy_adjoint()
timestep_function = lambda : timestep
checkpoints = {u: [], v: [], rho: []}

# Cost and gradient routines
def forward(vec):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(np.squeeze(vec), 2)
    u[weight_layout] = u0[weight_layout] + vec_split[0].reshape(u[weight_layout].shape)/local_M_matrix*np.sqrt(E0/0.5)
    v[weight_layout] = vec_split[1].reshape(v[weight_layout].shape)/local_M_matrix*np.sqrt(E0/0.5)
    rho['g'] = -0.5*erf(y/0.125)
    cost_t['g'] = 0
    # Evolve IVP
    checkpoints[u].clear()
    checkpoints[v].clear()
    checkpoints[rho].clear()
    solver.evolve(timestep_function=timestep_function,checkpoints=checkpoints)
    solve_psi.solve()
    cost = reducer.global_max(J['g'])
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
    return np.hstack([(cotangents[state][weight_layout]/local_M_matrix*np.sqrt(E0/0.5)).flatten() for state in [u,v]])

if test:
    # Taylor test
    u.fill_random('g');u['c'];u['g']
    v.fill_random('g');v['c'];v['g']
    vecp = np.hstack([(f['c']*local_M_matrix).flatten() for f in [u,v]])
    u.fill_random('g');u['c'];u['g']
    v.fill_random('g');v['c'];v['g']
    vec0 = np.hstack([(f['c']*local_M_matrix).flatten() for f in [u,v]])

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

    projected_gradient = comm.allreduce(np.vdot(grad0,vecp),op=MPI.SUM)
    first = np.abs(np.array(costs)-cost0)
    second = np.abs(np.array(costs)-cost0 - np.array(size)*projected_gradient)
    if comm.rank==0:
        logging.getLogger().setLevel(logging.INFO)
    logger.info('First order  : {0:f}'.format(linregress(np.log(size), np.log(first)).slope))
    logger.info('Second order : {0:f}'.format(linregress(np.log(size), np.log(second)).slope))
else:
    manifold = Sphere(2*Nx*Ny)

    @pymanopt.function.numpy(manifold)  
    def cost(vec):
        vecu, vecv = np.split(vec,2)
        local_vec = np.hstack([(v.reshape(gshape)[local_slice]).flatten() for v in [vecu,vecv]])
        return forward(local_vec)
    
    u_grad = u.copy_adjoint()
    v_grad = v.copy_adjoint()

    @pymanopt.function.numpy(manifold)
    def grad(vec):
        cost(vec)
        local_grad = backward()
        local_u, local_v = np.split(local_grad,2)
        u_grad[weight_layout] = local_u.reshape(u_grad[weight_layout].shape)
        v_grad[weight_layout] = local_v.reshape(v_grad[weight_layout].shape)
        grad = np.hstack([v.allgather_data(layout=weight_layout).flatten() for v in [u_grad,v_grad]])
        return grad
   
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

    if comm.rank==0:
        verbosity = 2
        log_verbosity = 1
    else:
        verbosity = 0
        log_verbosity = 0

    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity)
    sol = optimizer.run(problem)

    if comm.rank==0:
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]
        np.savez('convergence',iterations=iterations,costs=costs,gradient_norms=gradient_norms)

    # Get output for optimal
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt = 0.1, mode='overwrite')
    snapshots.add_tasks([u,v,rho])
    cost(sol.point)