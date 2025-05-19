import logging, time
import uuid
import numpy as np
import dedalus.public as d3
logger = logging.getLogger(__name__)

# Parameters
Nr = 64
n = 3 # Start n
ncc_cutoff = 1e-3
tolerance = 1e-10
dealias = 2
dtype = np.float64

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype)
ball = d3.BallBasis(coords, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)

# Fields
f = dist.Field(name='f', bases=ball)
tau = dist.Field(name='tau', bases=ball.surface)

X_list = []
dX_list = []
def cost_grad(n):
    # Problem and solver
    lift = lambda A: d3.Lift(A, ball, -1)
    problem = d3.NLBVP([f, tau], namespace=locals())
    problem.add_equation("lap(f) + lift(tau) = - f**n")
    problem.add_equation("f(r=1) = 0")
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

    # Solve NLBVP
    t_start = time.time()
    pert_norm = np.inf
    count = 0
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        count += 1
    logger.info('Time taken direct %f, number of Newton steps %d' % (time.time()-t_start, count))

    # Compute non-dimensional radius
    X = f(r=0)**((n-1)/2)

    # Compute sensitivity of X
    # Initialise cotangents
    Radj = X.evaluate().copy_adjoint()
    cotangents = {}
    Radj['g'] = 1
    cotangents[X] = Radj
    t_start = time.time()
    # Propagate cotangents through definition
    _, cotangents =  X.evaluate_vjp(cotangents, id=uuid.uuid4(), force=True)
    # Propagate cotagents through NLBVP
    cotangents = solver.compute_sensitivities(cotangents)
    logger.info('Time taken adjoint %f ' % (time.time()-t_start))
    # Use cotangents to compute sensitivity
    X_contribution = np.max((X*np.log(f(r=0))/2)['g'])
    f_contribution  = np.vdot((f**n*np.log(f))['g'], cotangents[problem.equations[0]['F']]['g']) 
    dXdn = X_contribution + f_contribution
    X_value = np.max(X['g'])
    logger.info('X = {0:f}, dXdn = {1:f}'.format(X_value, dXdn))
    
    # Save data
    X_list.append(X_value)
    dX_list.append(dXdn)
    return X_value, dXdn

# Initial guess
phi, theta, r = dist.local_grids(ball)
R0 = 3
f['g'] = R0**(2/(n-1)) * (1 - r**2)**2

# Solve NLBVP over range of n's
n_list = []
n = 3
rel_increase = 1.1
count = 0
while n<5:
    n_list.append(n)
    X_value, dXdn = cost_grad(n)
    dn = (rel_increase-1)*X_value/dXdn
    n += dn
    count += 1

# Save outputs
logger.info('Total number of iterations {0:d}'.format(count))
np.savez('Lane_Emden', n=n_list, R=np.array(X_list), dR=np.array(dX_list))
