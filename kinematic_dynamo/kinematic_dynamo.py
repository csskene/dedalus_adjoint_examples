"""

"""
import numpy as np
from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
import sys
from dedalus.extras.flow_tools import GlobalArrayReducer
import uuid
import scipy.sparse as sp

from mpi4py import MPI
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
reducer = GlobalArrayReducer(MPI.COMM_WORLD)
logger = logging.getLogger(__name__)

import pymanopt
from pymanopt.optimizers import ConjugateGradient

from pymanopt.manifolds.product import Product
from pymanopt.tools.diagnostics import check_gradient
logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Just for now
import sys
sys.path.append('../manifolds')
from Generalised_Stiefel import Generalised_Stiefel

# Parameters
N = 32
dealias = 1
timestep = 1e-3
timestepper = d3.RK222
NIter = int(1/timestep)

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')

# Choose mesh whose factors are most similar in size
factors = [[ncpu//i, i] for i in range(1, int(np.sqrt(ncpu))+1) if np.mod(ncpu, i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

dist = d3.Distributor(coords, dtype=np.float64, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# Fields
phi  = dist.Field(name='phi', bases=(xbasis, ybasis, zbasis))
u_init = dist.Field(name='u_init', bases=(xbasis, ybasis, zbasis))
v_init = dist.Field(name='v_init', bases=(xbasis, ybasis, zbasis))
w_init = dist.Field(name='w_init', bases=(xbasis, ybasis, zbasis))
u = dist.Field(name='u', bases=(xbasis, ybasis, zbasis))
v = dist.Field(name='v', bases=(xbasis, ybasis, zbasis))
w = dist.Field(name='w', bases=(xbasis, ybasis, zbasis))
Ax = dist.Field(name='Ax', bases=(xbasis, ybasis, zbasis))
Ay = dist.Field(name='Ay', bases=(xbasis, ybasis, zbasis))
Az = dist.Field(name='Az', bases=(xbasis, ybasis, zbasis))
Bx_init = dist.Field(name='Bx_init', bases=(xbasis, ybasis, zbasis))
By_init = dist.Field(name='By_init', bases=(xbasis, ybasis, zbasis))
Bz_init = dist.Field(name='Bz_init', bases=(xbasis, ybasis, zbasis))
Bx_clean = dist.Field(name='Bx_clean', bases=(xbasis, ybasis, zbasis))
By_clean = dist.Field(name='By_clean', bases=(xbasis, ybasis, zbasis))
Bz_clean = dist.Field(name='Bz_clean', bases=(xbasis, ybasis, zbasis))
Cx = dist.Field(name='Cx')
Cy = dist.Field(name='Cy')
Cz = dist.Field(name='Cz')
tau_phi = dist.Field(name='tau_phi')
Pi  = dist.Field(name='Pi', bases=(xbasis, ybasis, zbasis))
tau_Pi = dist.Field(name='tau_Pi')

# Substitutions
dx = lambda A: d3.Differentiate(A,coords['x'])
dy = lambda A: d3.Differentiate(A,coords['y'])
dz = lambda A: d3.Differentiate(A,coords['z'])
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
Bx = dy(Az) - dz(Ay)
By = dz(Ax) - dx(Az)
Bz = dx(Ay) - dy(Ax)
η = 0.5

# Global weight matrix (definitely better ways!)
weight = np.ones((N,N,N))*np.pi**3
# Adjust for zeroth rows
for i in range(N):
    for j in range(N):
        for k in range(N):
            p = np.count_nonzero(np.array([i, j, k])==0)
            weight[i, j, k] *= 2**p
weight /= (2*np.pi)**3
weight_layout = dist.coeff_layout
local_slice = weight_layout.slices(u.domain, scales=1)
gshape = weight_layout.global_shape(u.domain, scales=1)

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
solver.stop_iteration = NIter

# Problem for vector potential
problem_A = d3.LBVP([Ax, Ay, Az, phi, tau_phi, Cx, Cy, Cz], namespace=locals())
problem_A.add_equation("-lap(Ax) + dx(phi) + Cx = dy(Bz_clean) - dz(By_clean)")
problem_A.add_equation("-lap(Ay) + dy(phi) + Cy = dz(Bx_clean) - dx(Bz_clean)")
problem_A.add_equation("-lap(Az) + dz(phi) + Cz = dx(By_clean) - dy(Bx_clean)")
problem_A.add_equation("dx(Ax) + dy(Ay) + dz(Az) + tau_phi = 0")
problem_A.add_equation("integ(phi) = 0")
problem_A.add_equation("integ(Ax) = 0")
problem_A.add_equation("integ(Ay) = 0")
problem_A.add_equation("integ(Az) = 0")
solver_A = problem_A.build_solver()

# Problems for divergence and average cleaning
problem_u = d3.LBVP([u, v, w, Pi, tau_Pi, Cx, Cy, Cz], namespace=locals())
problem_u.add_equation("-u + dx(Pi) + Cx = -u_init")
problem_u.add_equation("-v + dy(Pi) + Cy = -v_init")
problem_u.add_equation("-w + dz(Pi) + Cz = -w_init")
problem_u.add_equation("dx(u) + dy(v) + dz(w) + tau_Pi = 0")
problem_u.add_equation("integ(Pi) = 0")
problem_u.add_equation("integ(u) = 0")
problem_u.add_equation("integ(v) = 0")
problem_u.add_equation("integ(w) = 0")
solver_u = problem_u.build_solver()

problem_B = d3.LBVP([Bx_clean, By_clean, Bz_clean, Pi, tau_Pi, Cx, Cy, Cz], namespace=locals())
problem_B.add_equation("-Bx_clean + dx(Pi) + Cx = -Bx_init")
problem_B.add_equation("-By_clean + dy(Pi) + Cy = -By_init")
problem_B.add_equation("-Bz_clean + dz(Pi) + Cz = -Bz_init")
problem_B.add_equation("dx(Bx_clean) + dy(By_clean) + dz(Bz_clean) + tau_Pi = 0")
problem_B.add_equation("integ(Pi) = 0")
problem_B.add_equation("integ(Bx_clean) = 0")
problem_B.add_equation("integ(By_clean) = 0")
problem_B.add_equation("integ(Bz_clean) = 0")
solver_B = problem_B.build_solver()

J = -d3.Average(Bx**2 + By**2 + Bz**2)
Jadj = J.evaluate().copy_adjoint()
timestep_function = lambda : timestep
checkpoints = {Ax: [], Ay: [], Az: []}

def forward(vecs):
    # Reset solver
    solver.reset()
    # Initial condition
    vec_split = np.split(vecs[0], 3)
    for i, vec in enumerate([u_init, v_init, w_init]):
        vec['c'] = vec_split[i].reshape(vec[weight_layout].shape)
    vec_split = np.split(vecs[1], 3)
    for i, vec in enumerate([Bx_init, By_init, Bz_init]):
        vec['c'] = vec_split[i].reshape(vec[weight_layout].shape)
    for key in list(checkpoints.keys()):
        checkpoints[key].clear()
    # Remove divergence and average
    solver_u.solve()
    solver_B.solve()
    # Solve for A
    solver_A.solve()
    # Evolve IVP and compute cost
    solver.evolve(timestep_function=timestep_function, checkpoints=checkpoints)
    cost = reducer.global_max(J['g'])
    return cost

def backward():
    Jadj['g'] = 1
    cotangents = {}
    cotangents[J] = Jadj
    _, cotangents =  J.evaluate_vjp(cotangents, id=uuid.uuid4(), force=True)
    cotangents = solver.compute_sensitivities(cotangents, checkpoints=checkpoints)
    cotangents = solver_A.compute_sensitivities(cotangents)
    cotangents = solver_u.compute_sensitivities(cotangents)
    cotangents = solver_B.compute_sensitivities(cotangents)
    grad_u = np.hstack([(cotangents[state]['c']).flatten() for state in [u_init, v_init, w_init]])
    grad_B = np.hstack([(cotangents[state]['c']).flatten() for state in [Bx_init, By_init, Bz_init]])
    return [grad_u, grad_B]

# Create the manifold
weight_sp = sp.diags(np.hstack([weight.flatten(), weight.flatten(), weight.flatten()]))
weight_inv = sp.diags(np.hstack([1/weight.flatten(), 1/weight.flatten(), 1/weight.flatten()]))
manifold_GS = Generalised_Stiefel(3*N**3, 1, weight_sp,Binv=weight_inv, k=1, retraction="polar")
manifold = Product([manifold_GS, manifold_GS])

# TODO: Tidy up parallelisation with scatters/gathers?
@pymanopt.function.numpy(manifold)  
def cost(vecU, vecB):
    vec = [vecU, vecB]
    vecu, vecv, vecw = np.split(vec[0],3)
    vecBx, vecBy, vecBz = np.split(vec[1],3)
    local_vec_U = np.hstack([(v.reshape(gshape)[local_slice]).flatten() for v in [vecu, vecv, vecw]])
    local_vec_B = np.hstack([(v.reshape(gshape)[local_slice]).flatten() for v in [vecBx, vecBy, vecBz]])
    return forward([local_vec_U,local_vec_B])

u_grad = u.copy_adjoint()
v_grad = v.copy_adjoint()
w_grad = w.copy_adjoint()

Bx_grad = Bx_init.copy_adjoint()
By_grad = By_init.copy_adjoint()
Bz_grad = Bz_init.copy_adjoint()
@pymanopt.function.numpy(manifold)
def grad(vecU,vecB):
    # Can comment cost evaluation if sure that optimiser always runs cost before gradient internally.
    # cost(vec)
    local_grad = backward()
    local_u, local_v, local_w = np.split(local_grad[0],3)
    u_grad[weight_layout] = local_u.reshape(u_grad[weight_layout].shape)
    v_grad[weight_layout] = local_v.reshape(v_grad[weight_layout].shape)
    w_grad[weight_layout] = local_w.reshape(w_grad[weight_layout].shape)
    gradU = np.hstack([v.allgather_data(layout=weight_layout).flatten() for v in [u_grad, v_grad, w_grad]])
    # Reshape to manifold shape
    gradU = gradU.reshape((len(gradU),1))

    local_Bx, local_By, local_Bz = np.split(local_grad[1],3)
    Bx_grad[weight_layout] = local_Bx.reshape(Bx_grad[weight_layout].shape)
    By_grad[weight_layout] = local_By.reshape(By_grad[weight_layout].shape)
    Bz_grad[weight_layout] = local_Bz.reshape(Bz_grad[weight_layout].shape)
    gradB = np.hstack([v.allgather_data(layout=weight_layout).flatten() for v in [Bx_grad, By_grad, Bz_grad]])
    # Reshape to manifold shape
    gradB = gradB.reshape((len(gradU),1))
    return [gradU, gradB]

# Parallel-safe random point and tangent-vector
random_point = manifold.random_point()
random_point = comm.bcast(random_point,root=0)
random_tangent_vector = manifold.random_tangent_vector(random_point)
random_tangent_vector = comm.bcast(random_tangent_vector,root=0)

verbosity = 2*(comm.rank==0)
log_verbosity = 1*(comm.rank==0)
problem_opt = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)
# check_gradient(problem_opt, x=random_point, d=random_tangent_vector)
# Random divergence-free, zero average initial condition
for field in [u_init, v_init, w_init, Bx_init, By_init, Bz_init]:
    field.fill_random(layout='c')
    field.low_pass_filter(scales=0.25)

solver_u.solve()
solver_B.solve()
initial_u = np.hstack([f.allgather_data(layout=weight_layout).flatten() for f in [u, v, w]]).reshape(*random_point[0].shape)
initial_B = np.hstack([f.allgather_data(layout=weight_layout).flatten() for f in [Bx_clean, By_clean, Bz_clean]]).reshape(*random_point[1].shape)

initial_u /= np.sqrt(initial_u.T@weight_sp@initial_u)
initial_B /= np.sqrt(initial_B.T@weight_sp@initial_B)

initial_point = [initial_u, initial_B]
optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity)
sol = optimizer.run(problem_opt, initial_point=initial_point)

if comm.rank==0:
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]
        np.savez('convergence', iterations=iterations, costs=costs, gradient_norms=gradient_norms, min_gradient_norm=1e-3)

# Get output for optimal seed
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt = 0.1, mode='overwrite')
snapshots.add_tasks([u, v, w])
snapshots.add_task(Bx, name='Bx')
snapshots.add_task(By, name='By')
snapshots.add_task(Bz, name='Bz')
cost(sol.point[0], sol.point[1])