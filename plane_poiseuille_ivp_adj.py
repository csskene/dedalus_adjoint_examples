"""

"""
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import numpy as np

from dedalus.tools import jacobi
import scipy.sparse as sp
import time

logger = logging.getLogger(__name__)

# Parameters
Ny = 128
dtype = np.complex128

alpha = 1
beta = 0
Re = 5000

# Bases and domain
coords = d3.CartesianCoordinates('y')
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.ChebyshevT(coords['y'], size=Ny,dealias=1, bounds=(0, 2))

y, = dist.local_grids(ybasis)

# Fields
u = dist.Field(name='u', bases=(ybasis))
v = dist.Field(name='v', bases=(ybasis))
w = dist.Field(name='w', bases=(ybasis))

p = dist.Field(name='p', bases=(ybasis))
tau_u_1 = dist.Field(name='tau_u_1')
tau_u_2 = dist.Field(name='tau_u_2')

tau_v_1 = dist.Field(name='tau_v_1')
tau_v_2 = dist.Field(name='tau_v_2')

tau_w_1 = dist.Field(name='tau_w_1')
tau_w_2 = dist.Field(name='tau_w_2')

tau_p = dist.Field(name='tau_p')

# Substitutions
lift_basis = ybasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

dy = lambda A: d3.Differentiate(A, coords['y'])

# Base flow
U = dist.Field(name='U',bases=(ybasis))
U['g'] = y*(2-y)
Uy = dy(U)

# Problem
problem = d3.IVP([u,v,w,p, tau_u_1, tau_u_2,tau_v_1, tau_v_2,tau_w_1, tau_w_2], namespace=locals())
problem.add_equation("dt(u) + 1j*alpha*u*U + v*Uy - 1/Re*(dy(dy(u))-alpha**2*u-beta**2*u) + lift(tau_u_1,-1) + lift(tau_u_2,-2) + 1j*alpha*p = 0")
problem.add_equation("dt(v) + 1j*alpha*v*U - 1/Re*(dy(dy(v))-alpha**2*v-beta**2*v) + lift(tau_v_1,-1) + lift(tau_v_2,-2) + dy(p) = 0")
problem.add_equation("dt(w) + 1j*alpha*w*U - 1/Re*(dy(dy(w))-alpha**2*w-beta**2*w) + lift(tau_w_1,-1) + lift(tau_w_2,-2) + 1j*beta*p = 0")
problem.add_equation("1j*alpha*u + dy(v) + 1j*beta*w = 0")

problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")

problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")

problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")

solver = problem.build_solver(d3.SBDF2)
# Get spectrally accurate weight matrices
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=(ybasis), adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
W = W_field['g']
# Cholesky decomposition
M = np.sqrt(W)
Minv = 1/M

def mult(vec,solver,Niter):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M 
    # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False

    vec_split = np.split(np.squeeze(vec), 3) 
    u['g'] = Minv*vec_split[0]
    v['g'] = Minv*vec_split[1]
    w['g'] = Minv*vec_split[2]
    
    solver.stop_iteration = Niter
    timestep = 0.5
    
    try:
        while solver.proceed:
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    grad = np.hstack((M*u['g'],M*v['g'],M*w['g']))

    return grad

def mult_hermitian(vec,solver,Niter):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H 
    solver.iteration = Niter
    solver.timestepper._LHS_params = False

    vec_split = np.split(np.squeeze(vec), 3)
    solver.state_adj[0]['g'] = M*vec_split[0]
    solver.state_adj[1]['g'] = M*vec_split[1]
    solver.state_adj[2]['g'] = M*vec_split[2]
    
    timestep = 0.5
    
    try:
        while solver.iteration>0:
            solver.step_adjoint(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
     
    grad = np.hstack((Minv*solver.state_adj[0]['g'],Minv*solver.state_adj[1]['g'],Minv*solver.state_adj[2]['g']))
    
    return grad

Phi = sp.linalg.LinearOperator((3*Ny,3*Ny),matvec= lambda A: mult(A,solver,100),rmatvec=lambda A: mult_hermitian(A,solver,100))

vec1 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)
vec2 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)

term2 = np.vdot(Phi.H@vec2,vec1)
term1 = np.vdot(vec2,Phi@vec1)

logger.info('Adjoint error = %g' % np.abs(term1-term2))

ts_tg = time.time()
gains = []
times = []
for niter in range(100)[1::5]:
    Phi = sp.linalg.LinearOperator((3*Ny,3*Ny),matvec= lambda A: mult(A,solver,niter),rmatvec=lambda A: mult_hermitian(A,solver,niter))
    ts = time.time()
    UH,sigma,V = sp.linalg.svds(Phi,k=1)
    logger.info('T = %f, Time taken for SVD = %f s' % (niter*0.5,time.time()-ts))

    gains.append(sigma[-1]**2)
    times.append(niter*0.5)

logger.info('Time taken for whole sweep %f' % (time.time()-ts_tg))

fig = plt.figure(figsize=(6, 4))
plt.plot(times,gains,'-.')
plt.ylabel("Gain")
plt.xlabel("T")
plt.title("Optimal gains for plane Poiseuille flow")
plt.savefig("plane_poiseuille_optimal_gains_transient_growth.png", dpi=200)

# plt.plot(y,Minv*UH[:Ny,-1].real)
# plt.plot(y,Minv*V[-1,:Ny].real)
# plt.show()