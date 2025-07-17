"""Compute sensitivity of the radius to the polytropic index in the Lane-Emden equation."""

import logging, time
import numpy as np
import dedalus.public as d3
from dedalus.tools import adjoint
logger = logging.getLogger(__name__)


# Parameters
Nr = 64
ncc_cutoff = 1e-3
tolerance = 1e-10
dealias = 2
dtype = np.float64

# Domain
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype)
ball = d3.BallBasis(coords, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)

# Fields
n = dist.Field(name='n')
f = dist.Field(name='f', bases=ball)
tau = dist.Field(name='tau', bases=ball.surface)

# Substitutions
lift = lambda A: d3.Lift(A, ball, -1)
R = f(r=0)**((n-1)/2)

# Problem
problem = d3.NLBVP([f, tau], namespace=locals())
problem.add_equation("lap(f) + lift(tau) = - f**n")
problem.add_equation("f(r=1) = 0")

# Cost and gradient function
def cost_grad(n_val):
    # Build solver
    n['g'] = n_val
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    # Forward solve
    pert_norm = np.inf
    t0 = time.time()
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f"Perturbation norm: {pert_norm:.3e}")
    t1 = time.time()
    logger.info(f"Forward solve: {solver.iteration} Newton steps")
    logger.info(f"Forward solve: {t1-t0} seconds")
    # Adjoint solve
    cotangents = adjoint.initialize_cotangents(R)
    cotangents = solver.accumulate_sensitivities(cotangents)
    t2 = time.time()
    logger.info(f"Adjoint solve: {t2-t1} seconds")
    R_val = R['g'][0,0,0]
    dRdn = cotangents[n]['g'][0,0,0]
    logger.info(f"R = {R_val:.3e}, dR/dn = {dRdn:.3e}")
    return R_val, dRdn

# Initial guess
phi, theta, r = dist.local_grids(ball)
R0 = 3
n0 = 3
f['g'] = R0**(2/(n0-1)) * (1 - r**2)**2

# Solve NLBVP over range of n's
n_list = []
R_list = []
dR_list = []
n_val = n0
R_step_rel = 0.1
while n_val < 5:
    print(n_val)
    R_val, dRdn = cost_grad(n_val)
    n_list.append(n_val)
    R_list.append(R_val)
    dR_list.append(dRdn)
    n_val += R_step_rel * R_val / dRdn

# Save outputs
logger.info(f"Total solves: {len(n_list)}")
np.savez('Lane_Emden', n=n_list, R=np.array(R_list), dR=np.array(dR_list))
