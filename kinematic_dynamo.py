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

# Choose optimisation package
optimisation_package = 'pymanopt'
# optimisation_package = 'SphereManOpt'
logger = logging.getLogger(__name__)

if optimisation_package =='pymanopt':
    import pymanopt
    from pymanopt.manifolds import Sphere
    from pymanopt.optimizers import ConjugateGradient
    from pymanopt.manifolds.product import Product

    from pymanopt.tools.diagnostics import check_gradient
    logging.getLogger().setLevel(logging.WARNING)
elif optimisation_package == 'SphereManOpt':
    sys.path.append('../SphereManOpt')
    from TestGrad import Adjoint_Gradient_Test
    from Sphere_Grad_Descent import Optimise_On_Multi_Sphere

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
N = 24
dealias = 3/2
timestep = 1e-3
timestepper = d3.SBDF1

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

phi  = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis,ybasis,zbasis))
B_init = dist.VectorField(coords, name='B_init', bases=(xbasis,ybasis,zbasis))
tau_phi = dist.Field(name='tau_phi')
V = dist.VectorField(coords, name='V')

x, y, z = dist.local_grids(xbasis,ybasis,zbasis)

# Substitutions
B = d3.curl(A)
J = -d3.lap(A)
η = 1

problem = d3.IVP([A, phi, tau_phi], namespace=locals())
problem.add_equation("dt(A) + grad(phi) - η*lap(A) = cross(u,B)")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")
solver = problem.build_solver(timestepper)

# For field to vector conversion
dom = A.domain
local_slice = dom.dist.coeff_layout.slices(dom,scales=1)
gshape = dom.dist.coeff_layout.global_shape(dom,scales=1)
lshape = dom.dist.coeff_layout.local_shape(dom,scales=1)
third = np.prod(gshape)
local_weight = weight[local_slice]

def vecToField(vec,v_field):
    """
    Takes vec and returns the field representation
    """
    # Ensure field in grid space with scales=1
    v_field['c']
    v_field['c'][0] = vec[:third].reshape(gshape)[local_slice]
    v_field['c'][1] = vec[third:2*third].reshape(gshape)[local_slice]
    v_field['c'][2] = vec[2*third:3*third].reshape(gshape)[local_slice]

def fieldToVec(B_field):
    """
    Takes x_field and returns the vector representation
    """
    vecB_phi   = np.zeros(gshape)
    vecB_theta = np.zeros(gshape)
    vecB_r     = np.zeros(gshape)
    # Make sure in grid space with scales=1
    B_field['c']
    B_field.change_scales(1)

    # The SphereManOpt method
    B_phi_global = comm.allgather(B_field['c'][0])
    B_theta_global = comm.allgather(B_field['c'][1])
    B_r_global = comm.allgather(B_field['c'][2])
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
###########################

# Problem for divergence cleaning
Pi  = dist.Field(name='Pi', bases=(xbasis,ybasis,zbasis))
tau_Pi = dist.Field(name='tau_Pi')
u_update  = dist.VectorField(coords, name='u_update', bases=(xbasis,ybasis,zbasis))

problem_u = d3.LBVP([Pi, tau_Pi], namespace=locals())
problem_u.add_equation("lap(Pi) + tau_Pi = div(u_update)")
problem_u.add_equation("integ(Pi) = 0")
solver_u = problem_u.build_solver()

# problem for vector potential
problem_A = d3.LBVP([A, phi, tau_phi, V], namespace=locals())
problem_A.add_equation("-lap(A) + grad(phi) + V = curl(B_init)")
problem_A.add_equation("div(A) + tau_phi = 0")
problem_A.add_equation("integ(phi) = 0")
problem_A.add_equation("integ(A) = 0")
solver_A = problem_A.build_solver()

states = []

def forward(vec):
    # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False

    vecToField(vec[0], u)
    vecToField(vec[1], B_init)
    u['c'][:] /= np.sqrt(local_weight)
    B_init['c'][:] /= np.sqrt(local_weight)

    solver_A.solve()

    solver.stop_iteration = NIter

    states.clear()
    A.change_scales(dealias)
    try:
        while solver.proceed:
            states.append(A['c'].copy())
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    B_cost = d3.curl(A).evaluate()['c']
    B_cost[:] *= np.sqrt(local_weight)

    cost = reducer.reduce_scalar(-np.linalg.norm(B_cost)**2,MPI.SUM)
    
    return cost

def backward(vec):
    solver.iteration = NIter
    solver.timestepper._LHS_params = False
    solver.timestepper.sensRHS.data *= 0

    solver.state_adj[0]['c'] = -2*(d3.curl(B))['c']
    solver.state_adj[0]['c'][:] *= local_weight
    
    count = -1
    try:
        while solver.iteration>0:
            A['c'] = states[count]
            count -= 1
            solver.step_adjoint(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    solver_A.state_adj[0]['c'] = solver.state_adj[0]['c']
    solver_A.solve_adjoint()

    # Divergence cleaning u
    u_update['c'] = solver.sens_adj[0]['c']
    solver_u.solve()
    u_grad_pre = (u_update - d3.grad(Pi))
    u_grad = (u_grad_pre - d3.integ(u_grad_pre)/(2*np.pi)**3).evaluate()
    u_grad['c'][:] /= np.sqrt(local_weight)

    gradU = fieldToVec(u_grad)

    solver.state_adj[0]['c'] = solver_A.F_adj[0]['c']
    solver.state_adj[0]['c'][:] /= np.sqrt(local_weight)
 
    gradA = fieldToVec(d3.curl(solver.state_adj[0]).evaluate())

    return [gradU, gradA]

def inner_product(x,y):
    return np.dot(x,y)

args_f = []
args_IP = []

# Random divergence free initial conditions
psi  = dist.VectorField(coords,name='psi', bases=(xbasis,ybasis,zbasis))
psi.fill_random(layout='c')
psi.low_pass_filter(scales=0.6)
A['c'] = d3.curl(psi)['c']
Ux0 = fieldToVec(A)

psi.fill_random(layout='c')
psi.low_pass_filter(scales=0.6)
u['c'] = d3.curl(psi)['c']
dUx0 = fieldToVec(u)

dUx0 /= np.sqrt(inner_product(dUx0,dUx0))
Ux0 /= np.sqrt(inner_product(Ux0,Ux0))

if optimisation_package =='pymanopt':
    manifold1 = Sphere(3*third)
    manifold2 = Sphere(3*third)

    manifold = Product([manifold1,manifold2])

    @pymanopt.function.numpy(manifold)  
    def cost(vecU,vecA):
        return forward([vecU,vecA])
    
    @pymanopt.function.numpy(manifold)
    def grad(vecU,vecA):
        forward([vecU,vecA])
        return backward([vecU,vecA])
   
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=grad)

    if rank==0:
        verbosity = 2
        log_verbosity = 1
    else:
        verbosity = 0
        log_verbosity = 0

    optimizer = ConjugateGradient(verbosity=verbosity, max_time=np.inf, max_iterations=100,  log_verbosity=log_verbosity)
    sol = optimizer.run(problem, initial_point = [Ux0,dUx0])

    if(rank==0):
        iterations     = optimizer._log["iterations"]["iteration"]
        costs          = optimizer._log["iterations"]["cost"]
        gradient_norms = optimizer._log["iterations"]["gradient_norm"]

        plot_convergence(iterations, costs, gradient_norms)

elif optimisation_package == 'SphereManOpt':
    cost = lambda A: forward(A)
    grad = lambda A: backward(A)
    LS = 'LS_wolfe'
    # LS = 'LS_armijo'

    Adjoint_Gradient_Test([Ux0,dUx0],[Ux0,dUx0], cost, grad, inner_product,args_f,args_IP,epsilon=1e-4)
    gradient_norms, costs, U_opt = Optimise_On_Multi_Sphere([Ux0,dUx0], [1,1],cost,grad,inner_product,args_f,args_IP, max_iters = 100, alpha_k = 100, LS=LS, CG=True)

    # iterations = np.linspace(1,len(costs),len(costs))

    # plot_convergence(iterations, -np.array(costs), np.linalg.norm(gradient_norms,axis=0))