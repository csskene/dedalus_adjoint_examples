"""
This script calculates the neutral curve for plane-Poiseuille flow

The neutral curve is computed as follows

1. Starting from a point, calculate the least stable eigenvalue and its
gradient with respect to the streamwise wavenumber alpha, and regularised
Reynolds number R=log_10(Re)/5 .
2. Perform a Newton solve in this direction of steepest ascent, to find
(alpha, R) that lies on the neutral curve.
3. From the point on the neutral curve, compute the tangent vector
and step in this direction to find a new starting point.
4. Repeat steps 1-3 to compute the neutral curve over a specified region.

The script should take less than 30 CPU seconds

Useage:
    python3 neutral_curve.py
"""
import dedalus.public as d3
import logging
import numpy as np
import scipy
from scipy.stats import linregress
logger = logging.getLogger(__name__)

# Parameters
Ny = 256
dtype = np.complex128
beta = 0

# Bases and domain
coords = d3.CartesianCoordinates('y')
dist   = d3.Distributor(coords, dtype=dtype)
ybasis = d3.Chebyshev(coords['y'], size=Ny, dealias=3/2, bounds=(0, 2))
y, = dist.local_grids(ybasis)
ybasis_k1 = ybasis.derivative_basis(1)
ybasis_k2 = ybasis.derivative_basis(2)
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
omega = dist.Field(name='omega')

# Adjoint fields
u_adj = dist.Field(name='u_adj', bases=(ybasis_k2))
v_adj = dist.Field(name='v_adj', bases=(ybasis_k2))
w_adj = dist.Field(name='w_adj', bases=(ybasis_k2))
p_adj = dist.Field(name='p_adj', bases=(ybasis_k2))

# Parameter fields
R     = dist.Field(name='R')
ten   = dist.Field(name='ten')
ten['g'] = 10
Re = ten**(5*R)
alpha = dist.Field(name='alpha')

# Substitutions
lift_basis = ybasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)
dt = lambda A: omega*A
dx = lambda A: 1j*alpha*A
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: 1j*beta*A
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))

# Base flow
U = dist.Field(name='U',bases=(ybasis))
U['g'] = y*(2-y)

# Problem
problem = d3.EVP([u, v, w, p, tau_u_1, tau_u_2, tau_v_1, tau_v_2, tau_w_1, tau_w_2], eigenvalue=omega, namespace=locals())
problem.add_equation("dt(u) + U*dx(u) + v*dy(U) - 1/Re*lap(u) + lift(tau_u_1,-1) + lift(tau_u_2,-2) + dx(p) = 0")
problem.add_equation("dt(v) + U*dx(v)           - 1/Re*lap(v) + lift(tau_v_1,-1) + lift(tau_v_2,-2) + dy(p) = 0")
problem.add_equation("dt(w) + U*dx(w)           - 1/Re*lap(w) + lift(tau_w_1,-1) + lift(tau_w_2,-2) + dz(p) = 0")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
# Boundary conditions
problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")
problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")
problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")
# Create solver
solver = problem.build_solver(ncc_cutoff=1e-10)

# Routine to compute the least stable eigenvalue and its gradient
def eig_grad(point, solver, target):
    R['g'] = point[0]
    alpha['g'] = point[1]
    # Solve the eigenvalue problem
    solver.solve_sparse(solver.subproblems[0], N=1, target=target, left=True, rebuild_matrices=True)
    index = np.argmax(solver.eigenvalues.real)
    # Compute sensitivities
    grad_Re = solver.compute_sensitivity(R, index, solver.subsystems[0])
    grad_alpha = solver.compute_sensitivity(alpha, index, solver.subsystems[0])
    cost = solver.eigenvalues[index]
    return cost, [grad_Re, grad_alpha]

# Routine to give the function and its gradient with respect to beta for the line
# search
def cost_grad_restricted(gamma, point0, normal, solver, target):
    point = point0 + normal*gamma
    eig, grad = eig_grad(point, solver, target)
    cost = eig.real
    cost_grad = np.array(grad).real
    beta_grad = np.dot(cost_grad, np.array(normal))
    return cost, beta_grad

# Test
point0 = np.array([np.log10(6500)/5, 1])
pointp = np.array([1, 0.01])
target = -0.25j
eig0, grad = eig_grad(point0, solver, target)
dir_grad = np.dot(grad, pointp)
residual = []
eps_list = []
eps = 1e-4
for kk in range(10):
    eps_list.append(eps)
    eig, grad = eig_grad(point0+eps*pointp, solver, target)
    residual.append(eig-eig0-eps*dir_grad)
    eps /= 2
residual = np.array(residual)
regression = linregress(np.log(eps_list), y=np.log(np.abs(residual.real)))
logger.info('Result of Taylor test (growth) %f' % (regression.slope))
regression = linregress(np.log(eps_list), y=np.log(np.abs(residual.imag)))
logger.info('Result of Taylor test (freq) %f' % (regression.slope))

# Main loop
points_0 = [[np.log10(6500)/5, 1], [1.56220763, 0.34540611]]
steps = [0.05, 0.01]
file_names = ['neutral', 'neutral_zoom']
lower_bounds = [0, 7/5]
for orig_point0, step_size, file_name, lower_bound in zip(points_0, steps, file_names, lower_bounds):
    neutral_p = []
    neutral_m = []
    # TODO: Could parallelise over dir
    for neutral, dir in zip([neutral_m, neutral_p], [-1, +1]):
        point0 = [0, 0]
        point0[0] = orig_point0[0]
        point0[1] = orig_point0[1]
        # Normals and tangents
        # Refine initial guess
        R['g'] = point0[0]
        alpha['g'] = point0[1]
        target = -0.01j
        solver.solve_sparse(solver.subproblems[0], N=10, target=target, rebuild_matrices=True)
        indices = np.argsort(solver.eigenvalues.real)
        index = indices[-1]
        target = solver.eigenvalues[index]
        while point0[0]<9/5 and point0[0]>lower_bound:
            eig, normal = eig_grad(point0, solver, target)
            normal = np.array(normal).real
            normal /= np.linalg.norm(normal)
            sol = scipy.optimize.root_scalar(lambda A: cost_grad_restricted(A, point0, normal, solver, target), x0=0, fprime=True)
            gamma = sol.root
            point = point0 + normal*gamma
            neutral.append(point)
            # Compute tangent
            eig, normal = eig_grad(point, solver, target)
            complex_normal = np.array(normal)
            normal = np.array(normal).real
            normal /= np.linalg.norm(normal)
            tangent = dir*np.array([-normal[1], normal[0]])
            dp = step_size*tangent
            point0 = point + dp
            target = eig + np.dot(complex_normal, dp)
            R['g'] = point0[0]
            alpha['g'] = point0[1]
    neutral = np.array(neutral_m[::-1] + neutral_p[1:])
    np.save(file_name, neutral)
