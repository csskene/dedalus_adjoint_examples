"""

"""
import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
import sys
from dedalus.extras.flow_tools import GlobalArrayReducer
import uuid
from scipy.stats import linregress

from mpi4py import MPI
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

# Choose optimisation package
# optimisation_package = 'pymanopt'
optimisation_package = 'SphereManOpt'
logger = logging.getLogger(__name__)

import pymanopt
from pymanopt.manifolds import Sphere
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds.product import Product

from pymanopt.tools.diagnostics import check_gradient
logging.getLogger().setLevel(logging.WARNING)

def plot_convergence(iterations, costs, gradient_norms):
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    ax1.plot(iterations, -np.array(costs),'C0',)
    ax2.semilogy(iterations, gradient_norms,'C1')
    plt.title(optimisation_package)
    ax1.set_xlabel(r'Iterations')
    ax1.set_ylabel(r'Cost', color='C0')
    ax2.set_ylabel(r'Gradient norm', color='C1')
    ax1.tick_params(axis='y', labelcolor='C0', color='C0')
    ax2.tick_params(axis='y', labelcolor='C1', color='C1')
    ax2.spines['right'].set_color('C1')
    ax2.spines['left'].set_color('C0')
    ax1.tick_params(axis='y', colors='C0', labelcolor='C0',which='both')
    ax2.tick_params(axis='y', colors='C1', labelcolor='C1',which='both')
    plt.savefig("kinematic_dynamo_{:s}.png".format(optimisation_package), dpi=200,bbox_inches='tight')

logger = logging.getLogger(__name__)

# Parameters
N = 32
dealias = 1
timestep = 1e-3
timestepper = d3.RK222

NIter = 1/timestep

# Global weight matrix (definitely better ways!)
weight = np.ones((N,N,N))*np.pi**3
# Adjust for zeroth rows
for i in range(N):
    for j in range(N):
        for k in range(N):
            p = np.count_nonzero(np.array([i,j,k])==0)
            weight[i,j,k] *= 2**p

weight /= (2*np.pi)**3
# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')

# Choose mesh whose factors are most similar in size
factors = [[ncpu//i,i] for i in range(1,int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh)

xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=N, bounds=(0, 2*np.pi), dealias=dealias)

x, y, z = dist.local_grids(xbasis,ybasis,zbasis)

# Fields
phi  = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
u = dist.Field(name='u', bases=(xbasis,ybasis,zbasis))
v = dist.Field(name='v', bases=(xbasis,ybasis,zbasis))
w = dist.Field(name='w', bases=(xbasis,ybasis,zbasis))
Ax = dist.Field(name='Ax', bases=(xbasis,ybasis,zbasis))
Ay = dist.Field(name='Ay', bases=(xbasis,ybasis,zbasis))
Az = dist.Field(name='Az', bases=(xbasis,ybasis,zbasis))
Bx_init = dist.Field(name='Bx_init', bases=(xbasis,ybasis,zbasis))
By_init = dist.Field(name='By_init', bases=(xbasis,ybasis,zbasis))
Bz_init = dist.Field(name='Bz_init', bases=(xbasis,ybasis,zbasis))
Vx = dist.Field(name='Vx')
Vy = dist.Field(name='Vy')
Vz = dist.Field(name='Vz')
tau_phi = dist.Field(name='tau_phi')

# Substitutions
dx = lambda A: d3.Differentiate(A,coords['x'])
dy = lambda A: d3.Differentiate(A,coords['y'])
dz = lambda A: d3.Differentiate(A,coords['z'])
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
Bx = dy(Az) - dz(Ay)
By = dz(Ax) - dx(Az)
Bz = dx(Ay) - dy(Ax)
η = 0.5

problem = d3.IVP([Ax, Ay, Az, phi, tau_phi, u, v, w], namespace=locals())
# problem.add_equation("dt(Ax) + dx(phi) - η*lap(Ax) = v*Bz - w*By")
# problem.add_equation("dt(Ay) + dy(phi) - η*lap(Ay) = w*Bx - u*Bz")
# problem.add_equation("dt(Az) + dz(phi) - η*lap(Az) = u*By - v*Bx")
# TODO: These two forms have different tapes
problem.add_equation("dt(Ax) + dx(phi) - η*lap(Ax) = v*(dx(Ay) - dy(Ax)) - w*(dz(Ax) - dx(Az))")
problem.add_equation("dt(Ay) + dy(phi) - η*lap(Ay) = w*(dy(Az) - dz(Ay)) - u*(dx(Ay) - dy(Ax))")
problem.add_equation("dt(Az) + dz(phi) - η*lap(Az) = u*(dz(Ax) - dx(Az)) - v*(dy(Az) - dz(Ay))")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")
problem.add_equation("dt(u) = 0")
problem.add_equation("dt(v) = 0")
problem.add_equation("dt(w) = 0")
solver = problem.build_solver(timestepper)
solver.stop_iteration = 100

# Problem for vector potential
problem_A = d3.LBVP([Ax, Ay, Az, phi, tau_phi, Vx, Vy, Vz], namespace=locals())
problem_A.add_equation("-lap(Ax) + dx(phi) + Vx = dy(Bz_init) - dz(By_init)")
problem_A.add_equation("-lap(Ay) + dy(phi) + Vy = dz(Bx_init) - dx(Bz_init)")
problem_A.add_equation("-lap(Az) + dz(phi) + Vz = dx(By_init) - dy(Bx_init)")
problem_A.add_equation("dx(Ax) + dy(Ay) + dz(Az) + tau_phi = 0")
problem_A.add_equation("integ(phi) = 0")
problem_A.add_equation("integ(Ax) = 0")
problem_A.add_equation("integ(Ay) = 0")
problem_A.add_equation("integ(Az) = 0")
solver_A = problem_A.build_solver()

J = 0.5*d3.integ(Bx**2 + By**2 + Bz**2)
Jadj = J.evaluate().copy_adjoint()
timestep_function = lambda : timestep
checkpoints = {Ax: [], Ay: [], Az: []}

def forward(vecs):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(vecs[0], 3)
    for i, vec in enumerate([u,v,w]):
        vec['c'] = vec_split[i].reshape((N,N,N))/np.sqrt(weight)
    vec_split = np.split(vecs[1], 3)
    for i, vec in enumerate([Bx_init,By_init,Bz_init]):
        vec['c'] = vec_split[i].reshape((N,N,N))/np.sqrt(weight)
    for key in list(checkpoints.keys()):
        checkpoints[key].clear()
    solver_A.solve()
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
    cotangents = solver.calculate_sensitivities(cotangents,checkpoints=checkpoints)
    cotangents = solver_A.compute_sensitivities(cotangents)
    grad_u = np.hstack([(cotangents[state]['c']/np.sqrt(weight)).flatten() for state in [u,v,w]])
    grad_B = np.hstack([(cotangents[state]['c']/np.sqrt(weight)).flatten() for state in [Bx_init,By_init,Bz_init]])
    return [grad_u,grad_B]

# Random IC
vecs = []
for i in range(2):
    Bx_init.fill_random(layout='g')
    Bx_init.low_pass_filter(scales=0.5)
    By_init.fill_random(layout='g')
    By_init.low_pass_filter(scales=0.5)
    Bz_init.fill_random(layout='g')
    Bz_init.low_pass_filter(scales=0.5)
    u['c'] = (dy(Bz_init) - dz(By_init))['c']
    v['c'] = (dz(Bx_init) - dx(Bz_init))['c']
    w['c'] = (dx(By_init) - dy(Bx_init))['c']
    norm = np.max(d3.integ(u**2+v**2+w**2)['g']) 
    u['c'] /= np.sqrt(norm)
    v['c'] /= np.sqrt(norm)
    w['c'] /= np.sqrt(norm)
    vecs.append(np.hstack([f['c'].flatten() for f in [u,v,w]]))
vecps = []
for i in range(2):
    Bx_init.fill_random(layout='c')
    Bx_init.low_pass_filter(scales=0.5)
    By_init.fill_random(layout='c')
    By_init.low_pass_filter(scales=0.5)
    Bz_init.fill_random(layout='c')
    Bz_init.low_pass_filter(scales=0.5)
    u['c'] = (dy(Bz_init) - dz(By_init))['c']
    v['c'] = (dz(Bx_init) - dx(Bz_init))['c']
    w['c'] = (dx(By_init) - dy(Bx_init))['c']
    norm = np.max(d3.integ(u**2+v**2+w**2)['g'])
    u['c'] /= np.sqrt(norm)
    v['c'] /= np.sqrt(norm)
    w['c'] /= np.sqrt(norm)
    vecps.append(np.hstack([f['c'].flatten() for f in [u,v,w]]))
cost0 = forward(vecs)
grad0 = backward()

eps = 1e-4
costs = []
size = []
for i in range(10):
    vecsp = [vecs[0]+eps*vecps[0],vecs[1]+eps*vecps[1]]
    costp = forward(vecsp)
    costs.append(costp)
    size.append(eps)
    eps /= 2

first = np.abs(np.array(costs)-cost0)
second = np.abs(np.array(costs)-cost0 - np.array(size)*np.vdot(grad0[0],vecps[0]) - np.array(size)*np.vdot(grad0[1],vecps[1]))
fig = plt.figure(figsize=(6, 4))
plt.loglog(size,first,label=r'First order')
plt.loglog(size,second,label=r'Second order')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Error')
plt.title(r'Taylor test')
plt.legend()
plt.show()
# plt.savefig('kinematic_dynamo_Taylor_test.png',dpi=200)

print('######## Taylor Test Results ########')
print('First order  : ',linregress(np.log(size), np.log(first)).slope)
print('Second order : ',linregress(np.log(size), np.log(second)).slope)
print('#####################################')
# def backward(vec):
#     solver.iteration = NIter
#     solver.timestepper._LHS_params = False
#     solver.timestepper.sensRHS.data *= 0

#     solver.state_adj[0]['c'] = -2*(d3.curl(B))['c']
#     solver.state_adj[0]['c'][:] *= local_weight
    
#     count = -1
#     try:
#         while solver.iteration>0:
#             A['c'] = states[count]
#             count -= 1
#             solver.step_adjoint(timestep)
#     except:
#         logger.error('Exception raised, triggering end of main loop.')
#         raise

#     solver_A.state_adj[0]['c'] = solver.state_adj[0]['c']
#     solver_A.solve_adjoint()

#     # Divergence cleaning u
#     u_update['c'] = solver.sens_adj[0]['c']
#     solver_u.solve()
#     u_grad_pre = (u_update - d3.grad(Pi))
#     u_grad = (u_grad_pre - d3.integ(u_grad_pre)/(2*np.pi)**3).evaluate()
#     u_grad['c'][:] /= np.sqrt(local_weight)

#     gradU = fieldToVec(u_grad)

#     solver.state_adj[0]['c'] = solver_A.F_adj[0]['c']
#     solver.state_adj[0]['c'][:] /= np.sqrt(local_weight)
 
#     gradA = fieldToVec(d3.curl(solver.state_adj[0]).evaluate())

#     return [gradU, gradA]

# def inner_product(x,y):
#     return np.dot(x,y)

# args_f = []
# args_IP = []

# # Random divergence free initial conditions
# psi  = dist.VectorField(coords,name='psi', bases=(xbasis,ybasis,zbasis))
# psi.fill_random(layout='c')
# psi.low_pass_filter(scales=0.6)
# A['c'] = d3.curl(psi)['c']
# Ux0 = fieldToVec(A)

# psi.fill_random(layout='c')
# psi.low_pass_filter(scales=0.6)
# u['c'] = d3.curl(psi)['c']
# dUx0 = fieldToVec(u)

# dUx0 /= np.sqrt(inner_product(dUx0,dUx0))
# Ux0 /= np.sqrt(inner_product(Ux0,Ux0))

# if optimisation_package =='pymanopt':
#     manifold1 = Sphere(3*third)
#     manifold2 = Sphere(3*third)

#     manifold = Product([manifold1,manifold2])

#     @pymanopt.function.numpy(manifold)  
#     def cost(vecU,vecA):
#         return forward([vecU,vecA])
    
#     @pymanopt.function.numpy(manifold)
#     def grad(vecU,vecA):
#         forward([vecU,vecA])
#         return backward([vecU,vecA])
   
#     problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

#     if rank==0:
#         verbosity = 2
#         log_verbosity = 1
#     else:
#         verbosity = 0
#         log_verbosity = 0

#     optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity)
#     sol = optimizer.run(problem, initial_point = [Ux0,dUx0])

#     if(rank==0):
#         iterations     = optimizer._log["iterations"]["iteration"]
#         costs          = optimizer._log["iterations"]["cost"]
#         gradient_norms = optimizer._log["iterations"]["gradient_norm"]

#         plot_convergence(iterations, costs, gradient_norms)

# elif optimisation_package == 'SphereManOpt':
#     cost = lambda A: forward(A)
#     grad = lambda A: backward(A)
#     LS = 'LS_wolfe'
#     # LS = 'LS_armijo'

#     Adjoint_Gradient_Test([Ux0,dUx0],[Ux0,dUx0], cost, grad, inner_product,args_f,args_IP,epsilon=1e-4)
#     gradient_norms, costs, U_opt = Optimise_On_Multi_Sphere([Ux0,dUx0], [1,1],cost,grad,inner_product,args_f,args_IP, max_iters = 100, alpha_k = 100, LS=LS, CG=True)

#     # iterations = np.linspace(1,len(costs),len(costs))

#     # plot_convergence(iterations, -np.array(costs), np.linalg.norm(gradient_norms,axis=0))