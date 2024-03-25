import numpy as np

from dedalus import public as d3
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=32, bounds=(0, 2*np.pi), dealias=1)
ybasis = d3.RealFourier(coords['y'], size=32, bounds=(0, 2*np.pi), dealias=1)
zbasis = d3.RealFourier(coords['z'], size=32, bounds=(0, 2*np.pi), dealias=1)

phi  = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis,ybasis,zbasis))
tau_phi = dist.Field(name='tau_phi')

x, y, z = dist.local_grids(xbasis,ybasis,zbasis)

u['g'] = np.cos(x)
# Substitutions
B = d3.curl(A)
J = -d3.lap(A)
η = 0.5

problem = d3.IVP([A, phi, tau_phi], namespace=locals())
problem.add_equation("dt(A) + grad(phi) - η*lap(A) = d3.cross(u,B)")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")
solver = problem.build_solver(d3.SBDF1)
c_shape = u['g'].shape
# print(c_shape)

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
NN = 10
def cost(vec,solver):
    # solver.iteration = 0
    # problem.time['g'] = 0 
    # solver.sim_time = 0
    # solver.initial_iteration = 0
    # # reset evaluator
    # solver.evaluator.handlers[0].last_wall_div = -1
    # solver.evaluator.handlers[0].last_sim_div = -1
    # solver.evaluator.handlers[0].last_iter_div = -1
    # # reset timestepper
    # solver.timestepper._iteration = 0
    
    # # Reset solver
    solver.iteration = 0
    solver.initial_iteration = 0
    solver.evaluator.handlers[0].last_iter_div = -1
    # reset timestepper
    solver.timestepper._iteration = 0
    solver.timestepper._LHS_params = False

    A['g'][2] = np.cos(x)
    A['g'][1] = 0
    A['g'][0] = 0

    u['g'] = vec.reshape(c_shape)
    
    solver.stop_iteration = NN
    timestep = 0.01

    states.clear()
    
    try:
        while solver.proceed:
            states.append(A['g'].copy())
            solver.step(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise

    cost = np.linalg.norm(B['g'])**2
    print(cost)
    return cost

def grad(solver):
    solver.iteration = NN
    solver.timestepper._LHS_params = False

    solver.state_adj[0]['g'] = 2*d3.curl(B)['g'].reshape(c_shape)
    
    timestep = 0.01
    count = -1
    try:
        while solver.iteration>0:
            A['g'] = states[count]
            count -= 1
            solver.step_adjoint(timestep)
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
     
    grad = (solver.sens_adj[0]['g']).flatten()
    
    return grad


# Taylor test

forcing0 = np.random.rand(np.prod(c_shape))
forcingp = np.random.rand(np.prod(c_shape))

cost0 = cost(forcing0,solver)
grad0 = grad(solver)

eps = 0.001
costs = []
size = []
for i in range(10):
    costp = cost(forcing0+eps*forcingp,solver)
    costs.append(costp)
    size.append(eps)
    eps /= 2

first = np.abs(np.array(costs)-cost0)
second = np.abs(np.array(costs)-cost0 - np.array(size)*np.vdot(grad0,forcingp))
print((np.array(costs)-cost0)/np.array(size),np.vdot(grad0,forcingp))
plt.loglog(size,first,label=r'First order')
plt.loglog(size,second,label=r'Second order')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Taylor test')
plt.legend()
plt.show()

from scipy.stats import linregress
print('######## Taylor Test Results ########')
print('First order  : ',linregress(np.log(size), np.log(first)).slope)
print('Second order : ',linregress(np.log(size), np.log(second)).slope)
print('#####################################')

