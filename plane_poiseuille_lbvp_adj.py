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

alpha = -1
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

# # Forcing
fu = dist.Field(name='fu',bases=(ybasis))
fv = dist.Field(name='fv',bases=(ybasis))
fw = dist.Field(name='fw',bases=(ybasis))

omega = dist.Field(name='omega')
# Problem
problem = d3.LBVP([u,v,w,p, tau_u_1, tau_u_2,tau_v_1, tau_v_2,tau_w_1, tau_w_2], namespace=locals())
problem.add_equation("1j*omega*u + 1j*alpha*u*U + v*Uy - 1/Re*(dy(dy(u))-alpha**2*u-beta**2*u) + lift(tau_u_1,-1) + lift(tau_u_2,-2) + 1j*alpha*p = fu")
problem.add_equation("1j*omega*v + 1j*alpha*v*U - 1/Re*(dy(dy(v))-alpha**2*v-beta**2*v) + lift(tau_v_1,-1) + lift(tau_v_2,-2) + dy(p) = fv")
problem.add_equation("1j*omega*w + 1j*alpha*w*U - 1/Re*(dy(dy(w))-alpha**2*w-beta**2*w) + lift(tau_w_1,-1) + lift(tau_w_2,-2) + 1j*beta*p = fw")
problem.add_equation("1j*alpha*u + dy(v) + 1j*beta*w = 0")

problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")

problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")

problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")

solver = problem.build_solver()
# Get spectrally accurate weight matrices
a_, b_ = ybasis.a, ybasis.b
W_field = dist.Field(name='W_field', bases=(ybasis), adjoint=True)
W_field['c'] = jacobi.integration_vector(Ny, a_, b_)
W = W_field['g']
# Cholesky decomposition
M = np.sqrt(W)
Minv = 1/M

def mult(vec,solver):
    # Modified resolvent matrix is R_M = (M R Minv)
    # This function multiplies by R_M 

    vec_split = np.split(np.squeeze(vec), 3) 
    fu['g'] = Minv*vec_split[0]
    fv['g'] = Minv*vec_split[1]
    fw['g'] = Minv*vec_split[2]
    
    solver.solve()

    grad = np.hstack((M*u['g'],M*v['g'],M*w['g']))

    return grad

def mult_hermitian(vec,solver):
    # Modified state transition matrix is Phi_M = (M Phi Minv)
    # This function multiplies by Phi_M^H 

    vec_split = np.split(np.squeeze(vec), 3)
    solver.dJdX[0]['g'] = M*vec_split[0]
    solver.dJdX[1]['g'] = M*vec_split[1]
    solver.dJdX[2]['g'] = M*vec_split[2]
    
    dJdf = solver.solve_adjoint([fu,fv,fw])

    grad = np.hstack((Minv*dJdf[0]['g'],Minv*dJdf[1]['g'],Minv*dJdf[2]['g']))
    
    return grad

R = sp.linalg.LinearOperator((3*Ny,3*Ny),matvec= lambda A: mult(A,solver),rmatvec=lambda A: mult_hermitian(A,solver))

vec1 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)
vec2 = np.random.rand(Ny*3) + 1j*np.random.rand(Ny*3)

term2 = np.vdot(R.H@vec2,vec1)
term1 = np.vdot(vec2,R@vec1)

logger.info('Adjoint error = %g' % np.abs(term1-term2))

ts_tg = time.time()
gains = []
omegas = np.linspace(0.1,1.5,120)
for om in omegas:
    omega['g'] = om
    solver = problem.build_solver()
    Phi = sp.linalg.LinearOperator((3*Ny,3*Ny),matvec= lambda A: mult(A,solver),rmatvec=lambda A: mult_hermitian(A,solver))
    ts = time.time()
    UH,sigma,V = sp.linalg.svds(Phi,k=1)
    logger.info('om = %f, Time taken for SVD = %f s' % (om,time.time()-ts))

    gains.append(sigma[-1]**2)

logger.info('Time taken for whole sweep %f' % (time.time()-ts_tg))

fig = plt.figure(figsize=(6, 4))
plt.semilogy(omegas,gains,'-.')
plt.xlabel("Gain")
plt.ylabel("$\omega$")
plt.title("Optimal gains for plane Poiseuille flow (forced)")
# plt.show()
plt.savefig("plane_poiseuille_optimal_gains_resolvent.png", dpi=200)

# plt.plot(y,Minv*UH[:Ny,-1].real)
# plt.plot(y,Minv*V[-1,:Ny].real)
# plt.show()