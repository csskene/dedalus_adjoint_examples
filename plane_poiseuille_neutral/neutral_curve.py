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

Useage:
    python3 neutral_curve.py
"""
import dedalus.public as d3
import logging
import numpy as np
import scipy
logger = logging.getLogger(__name__)

def set_state_adjoint(self, index, subsystem):
    """
    Set state vector to the specified eigenmode.
    Parameters
    ----------
    index : int
        Index of desired eigenmode.
    subsystem : Subsystem object or int
        Subsystem that will be set to the corresponding eigenmode.
        If an integer, the corresponding subsystem of the last specified
        eigenvalue_subproblem will be used.
    """
    # TODO: allow setting left modified eigenvectors?
    subproblem = self.eigenvalue_subproblem
    if isinstance(subsystem, int):
        subsystem = subproblem.subsystems[subsystem]
    # Check selection
    if subsystem not in subproblem.subsystems:
        raise ValueError("subsystem must be in eigenvalue_subproblem")
    # Set coefficients
    for var in self.state:
        var['c'] = 0
    subsystem.scatter(self.left_eigenvectors[:, index], self.state)

# Parameters
Ny = 128
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
dy = lambda A: d3.Differentiate(A, coords['y'])

# Base flow
U = dist.Field(name='U',bases=(ybasis))
U['g'] = y*(2-y)
Uy = dy(U)

# Problem
problem = d3.EVP([u, v, w, p, tau_u_1, tau_u_2, tau_v_1, tau_v_2, tau_w_1, tau_w_2], eigenvalue=omega, namespace=locals())
problem.add_equation("omega*u + 1j*alpha*u*U + v*Uy - 1/Re*(dy(dy(u))-alpha**2*u-beta**2*u) + lift(tau_u_1,-1) + lift(tau_u_2,-2) + 1j*alpha*p = 0")
problem.add_equation("omega*v + 1j*alpha*v*U - 1/Re*(dy(dy(v))-alpha**2*v-beta**2*v) + lift(tau_v_1,-1) + lift(tau_v_2,-2) + dy(p) = 0")
problem.add_equation("omega*w + 1j*alpha*w*U - 1/Re*(dy(dy(w))-alpha**2*w-beta**2*w) + lift(tau_w_1,-1) + lift(tau_w_2,-2) + 1j*beta*p = 0")
problem.add_equation("1j*alpha*u + dy(v) + 1j*beta*w = 0")
# Boundary conditions
problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=2) = 0")
problem.add_equation("v(y=0) = 0")
problem.add_equation("v(y=2) = 0")
problem.add_equation("w(y=0) = 0")
problem.add_equation("w(y=2) = 0")
# Create solver
solver = problem.build_solver(ncc_cutoff=1e-10)

# Create parametric sensitivity gradients
dLudRe = d3.Convert(problem.eqs[0]['L'].sym_diff(R), ybasis_k2)
dLvdRe = d3.Convert(problem.eqs[1]['L'].sym_diff(R), ybasis_k2)
dLwdRe = d3.Convert(problem.eqs[2]['L'].sym_diff(R), ybasis_k2)

dLudalpha = d3.Convert(problem.eqs[0]['L'].sym_diff(alpha), ybasis_k2)
dLvdalpha = d3.Convert(problem.eqs[1]['L'].sym_diff(alpha), ybasis_k2)
dLwdalpha = d3.Convert(problem.eqs[2]['L'].sym_diff(alpha), ybasis_k2)
dLpdalpha = d3.Convert(problem.eqs[3]['L'].sym_diff(alpha), ybasis_k1)

# Routine to compute the least stable eigenvalue and its gradient
def eig_grad(point, solver, target):
    R['g'] = point[0]
    alpha['g'] = point[1]
    # Get point
    solver.solve_sparse(solver.subproblems[0], N=1, target=target, left=True, rebuild_matrices=True)
    # print(target, solver.eigenvalues[0])
    index = np.argmax(solver.eigenvalues.real)
    # Set the adjoint_state
    set_state_adjoint(solver, index, solver.subsystems[0])
    for field, adjoint_field in zip([u, v, w, p], [u_adj, v_adj, w_adj, p_adj]):
        adjoint_field['c'] = field['c']
    solver.set_state(index, solver.subsystems[0])
    # Get the gradient
    grad_Re = 0
    for adjoint_field, field in zip([u_adj, v_adj, w_adj], [dLudRe, dLvdRe, dLwdRe]):
        grad_Re -= np.vdot(adjoint_field['c'], field['c'])

    grad_alpha = 0
    for adjoint_field, field in zip([u_adj, v_adj, w_adj, p_adj], [dLudalpha, dLvdalpha, dLwdalpha, dLpdalpha]):
        grad_alpha -= np.vdot(adjoint_field['c'], field['c'])

    cost = solver.eigenvalues[index]
    return cost, [grad_Re, grad_alpha]

# Routine to give the function and its gradient with respect to beta for the line
# search
def cost_grad_restricted(beta, point0, normal, solver, target):
    point = point0 + normal*beta
    eig, grad = eig_grad(point, solver, target)
    cost = eig.real
    cost_grad = np.array(grad).real
    beta_grad = np.dot(cost_grad, normal)
    return cost, beta_grad

neutral_p = []
neutral_m = []
# Main loop
# TODO: Could parallelise over dir
for neutral, dir in zip([neutral_m, neutral_p], [-1, +1]):
    # Normals and tangents
    point0 = [np.log10(6500)/5, 0.8]
    # Refine initial guess
    R['g'] = point0[0]
    alpha['g'] = point0[1]
    target = 0.001715391802888724-0.2529293654585668j
    solver.solve_sparse(solver.subproblems[0], N=10, target=target, rebuild_matrices=True)
    index = np.argmax(solver.eigenvalues.real)
    target = solver.eigenvalues[index]
    while point0[0]<8/5:
        # Find neutral curve
        eig, normal = eig_grad(point0, solver, target)
        normal = np.array(normal).real
        normal /= np.linalg.norm(normal)
        sol = scipy.optimize.root_scalar(lambda A: cost_grad_restricted(A, point0, normal, solver, target), fprime=True, x0=0, method='newton')
        beta_ = sol.root
        point = point0 + normal*beta_
        neutral.append(point)
        # Compute tangent
        eig, normal = eig_grad(point, solver, target)    
        complex_normal = np.array(normal)
        normal = np.array(normal).real
        normal /= np.linalg.norm(normal)
        tangent = dir*np.array([-normal[1], normal[0]])
        
        dp = 0.05*tangent
        point0 = point + dp

        target = eig + np.dot(complex_normal, dp)
        R['g'] = point0[0]
        alpha['g'] = point0[1]

neutral = neutral_m[::-1]
neutral.extend(neutral_p[1:])
neutral = np.array(neutral)
np.save('neutral', neutral)
