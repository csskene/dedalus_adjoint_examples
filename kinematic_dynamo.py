import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt
import sys
from dedalus.extras.flow_tools import GlobalArrayReducer

from mpi4py import MPI
comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

sys.path.append('../SphereManOpt')
from TestGrad import Adjoint_Gradient_Test
from Sphere_Grad_Descent import Optimise_On_Multi_Sphere

logger = logging.getLogger(__name__)

# Parameters
N = 24
dealias = 1
timestep = 1e-3
timestepper = d3.SBDF1

NIter = 1/timestep

# LS = 'LS_wolfe'
LS = 'LS_armijo'

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.float64,mesh=[2,2])

xbasis = d3.RealFourier(coords['x'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=N, bounds=(0, 2*np.pi), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=N, bounds=(0, 2*np.pi), dealias=dealias)

factor = 1/N**3

phi  = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis,ybasis,zbasis))
tau_phi = dist.Field(name='tau_phi')

x, y, z = dist.local_grids(xbasis,ybasis,zbasis)

# Substitutions
B = d3.curl(A)
J = -d3.lap(A)
η = 1

problem = d3.IVP([A, phi, tau_phi], namespace=locals())
problem.add_equation("dt(A) + grad(phi) - η*lap(A) = d3.cross(u,B)")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")
solver = problem.build_solver(timestepper)

# For field to vector conversion
dom = A.domain
local_slice = dom.dist.grid_layout.slices(dom,scales=1)
gshape = dom.dist.grid_layout.global_shape(dom,scales=1)
third = np.prod(gshape)

def vecToField(vec,v_field):
    """
    Takes vec and returns the field representation
    """
    # Ensure field in grid space with scales=1
    v_field['g']
    v_field.change_scales(1)
    v_field['g'][0] = vec[:third].reshape(gshape)[local_slice]
    v_field['g'][1] = vec[third:2*third].reshape(gshape)[local_slice]
    v_field['g'][2] = vec[2*third:3*third].reshape(gshape)[local_slice]

def fieldToVec(B_field):
    """
    Takes x_field and returns the vector representation
    """
    vecB_phi   = np.zeros(gshape)
    vecB_theta = np.zeros(gshape)
    vecB_r     = np.zeros(gshape)
    # Make sure in grid space with scales=1
    B_field['g']
    B_field.change_scales(1)

    # The SphereManOpt method
    B_phi_global = comm.allgather(B_field['g'][0])
    B_theta_global = comm.allgather(B_field['g'][1])
    B_r_global = comm.allgather(B_field['g'][2])
    G_slices = comm.allgather(local_slice)
    for i in range(ncpu):
        vecB_phi[G_slices[i]] = np.real(B_phi_global[i])
        vecB_theta[G_slices[i]] = np.real(B_theta_global[i])
        vecB_r[G_slices[i]] = np.real(B_r_global[i])
    
    vec = np.hstack((vecB_phi.reshape(third),vecB_theta.reshape(third),vecB_r.reshape(third)))
    return vec

# Manually set adjoint RHS #
adjoint_terms = [-d3.curl(d3.cross(u,solver.state_adj[0])), problem.eqs[1]['F'],  problem.eqs[2]['F']]
F_adjoint_handler = solver.evaluator.add_system_handler(iter=1, group='F_adjoint')
for eq in adjoint_terms:
    F_adjoint_handler.add_task(eq)
F_adjoint_handler.build_system()
solver.F_adjoint = F_adjoint_handler.fields
for f in solver.F_adjoint:
    f.adjoint=True
############################
# sensitivity handler
sens_terms = [d3.cross(B,solver.state_adj[0]), problem.eqs[1]['F'],  problem.eqs[2]['F']]
F_sens_handler = solver.evaluator.add_system_handler(iter=1, group='F_sens')
for eq in sens_terms:
    F_sens_handler.add_task(eq)
F_sens_handler.build_system()
solver.F_sens = F_sens_handler.fields
for f in solver.F_sens:
    f.adjoint=True
############################
    
states = []

def cost(vec):
    # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False

    vecToField(vec[0],u)
    vecToField(vec[1],A)

    solver.stop_iteration = NIter

    states.clear()
    
    try:
        while solver.proceed:
            states.append(A['c'].copy())
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    cost = reducer.reduce_scalar(-np.linalg.norm(B['g'])**2*factor,MPI.SUM)
    if rank==0:
        print(-cost)
    return cost

def grad(vec):
    solver.iteration = NIter
    solver.timestepper._LHS_params = False
    solver.timestepper.sensRHS.data *= 0
    solver.state_adj[0].change_scales(dealias)
    solver.state_adj[0]['g'] = -2*d3.curl(B)['g']*factor
    
    count = -1
    try:
        while solver.iteration>0:
            A['c'] = states[count]
            count -= 1
            solver.step_adjoint(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    gradU = fieldToVec(solver.sens_adj[0])
    gradA = fieldToVec(solver.state_adj[0])
    
    return [gradU,gradA]

def inner_product(x,y):
    return np.dot(x,y)

args_f = []
args_IP = []

psi  = dist.VectorField(coords,name='psi', bases=(xbasis,ybasis,zbasis))
psi.fill_random()
psi.low_pass_filter(scales=0.1)
A.change_scales(dealias)
A['g'] = d3.curl(psi)['g']
Ux0 = fieldToVec(A)
psi.fill_random()
psi.low_pass_filter(scales=0.1)
u.change_scales(dealias)
u['g'] = d3.curl(psi)['g']
dUx0 = fieldToVec(u)

RESIDUAL,FUNCT,U_opt = Optimise_On_Multi_Sphere([Ux0,dUx0], [1/factor,1/factor],cost,grad,inner_product,args_f,args_IP, err_tol = 1e-06, max_iters = 10, alpha_k = 10000, LS=LS, CG=True)

