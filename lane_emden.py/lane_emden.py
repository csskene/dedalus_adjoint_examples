import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Nr = 64
n = 2
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
# Substitutions
lift = lambda A: d3.Lift(A, ball, -1)

# Initial guess
phi, theta, r = dist.local_grids(ball)
R0 = 5
f['g'] = R0**(2/(n-1)) * (1 - r**2)**2

R_list = []
dR_list = []
M_list = []
def cost_grad(n):
    # Solver
    lift = lambda A: d3.Lift(A, ball, -1)
    problem = d3.NLBVP([f, tau], namespace=locals())
    problem.add_equation("lap(f) + lift(tau) = - f**n")
    problem.add_equation("f(r=1) = 0")
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    steps = [f['g'].ravel().copy()]
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        f0 = f(r=0)
        Ri = f0**((n-1)/2)
        R_iterate = np.max(Ri['g'])
        logger.info(f'R iterate: {R_iterate}')
        steps.append(f['g'].ravel().copy())
    # Radius
    R = f(r=0)**((n-1)/2)
    Radj = R.evaluate().copy_adjoint()
    cotangents = {}
    Radj['g'] = 1
    cotangents[Ri] = Radj
    _, cotangents =  Ri.evaluate_vjp(cotangents, id=id, force=True)
    cotangents = solver.compute_sensitivities(cotangents)
    R_contribution = np.max((R*np.log(f(r=0))/2)['g'])
    f_contribution  = np.vdot((f**n*np.log(f))['g'], cotangents[problem.equations[0]['F']]['g']) 
    dRdn = R_contribution + f_contribution
    R_value = np.max(R['g'])
    logger.info('R = {0:f}, dRdn = {1:f}'.format(R_value, dRdn))
    R_list.append(R_value)
    dR_list.append(dRdn)
    # TODO: Mass

n_list = np.linspace(2, 3, 5)
for n in n_list:
    cost_grad(n)