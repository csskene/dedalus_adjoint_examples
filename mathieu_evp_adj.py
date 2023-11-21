"""
Dedalus script showing the use of adjoints for interpolating the eigenvalues of 
the Mathieu equation. This script is based on the d3 example and runs a coarse 
solve using adjoints for interpolation, and a fine solve for comparison.
It should take just a few seconds to run (serial only).

This script 

We use a Fourier basis to solve the EVP:
    dx(dx(y)) + (a - 2*q*cos(2*x))*y = 0
where 'a' is the eigenvalue. Periodicity is enforced by using the Fourier basis.

The gradient da/dq is found using the left eigenvectors.

To run and plot:
    $ python3 mathieu_evp_adj.py
"""
# Helper functions which could maybe be removed
def cubeFit(x1,x2,fx1,fx2,fx1d,fx2d):
    # Taken from linear analysis tools repo
    # Finds coefficients f(x) = Ax^3 + Bx^2 + Cx + D such that
    # f(x1)  = fx1
    # f(x2)  = fx2
    # f'(x1) = fx1d
    # f'(x2) = fx2d

    RHS = np.array([fx1,fx2,fx1d,fx2d]);
    row1 = np.array([x1**3, x1**2, x1, 1])
    row2 = np.array([x2**3, x2**2, x2, 1])
    row3 = np.array([3*x1**2, 2*x1, 1, 0])
    row4 = np.array([3*x2**2, 2*x2, 1, 0])
    LHS = np.vstack((row1,row2,row3,row4))
    ANS = np.linalg.solve(LHS,RHS)
    A = ANS[0]
    B = ANS[1]
    C = ANS[2]
    D = ANS[3]
    return A,B,C,D

# Function to get adjoint eigenvector (as its not in Dedalus yet)
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
    subsystem.scatter(self.modified_left_eigenvectors[:, index], self.state)

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import time
logger = logging.getLogger(__name__)

# Parameters
N = 32
q_list = np.linspace(0, 30, 10)
q_list_f = np.linspace(0, 30, 100)
# Basis
coord = d3.Coordinate('x')
dist = d3.Distributor(coord, dtype=np.complex128)
basis = d3.ComplexFourier(coord, N, bounds=(0, 2*np.pi))

# Fields
y = dist.Field(bases=basis)
a = dist.Field()

# Substitutions
x = dist.local_grid(basis)
q = dist.Field()
cos_2x = dist.Field(bases=basis)
cos_2x['g'] = np.cos(2 * x)
dx = lambda A: d3.Differentiate(A, coord)

# Problem
problem = d3.EVP([y], eigenvalue=a, namespace=locals())
problem.add_equation("dx(dx(y)) + (a - 2*q*cos_2x)*y = 0")

# Solver
solver = problem.build_solver()
evals = []
grads = []

# Coarse eigenvalue sweep with gradient calculation
tc = time.time()
for qi in q_list:
    q['g'] = qi
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True,left=True)
    sorted_evals = np.sort(solver.eigenvalues.real)
    indices = np.argsort(solver.eigenvalues.real)

    # Use the left eigenvectors to calculate the gradients
    sub_grad=[]
    dLdq = 2*cos_2x

    for index in range(10):
        solver.set_state(indices[index],solver.subsystems[0])
        y_dir = y.copy()
        set_state_adjoint(solver,indices[index],solver.subsystems[0])
        y_adj = y.copy()

        grad = np.vdot(y_adj['c'],(dLdq*y_dir)['c'])
        sub_grad.append(grad)
    
    evals.append(sorted_evals[:10])
    grads.append(sub_grad)
tc = time.time() - tc

# Fine eigenvalue sweep without gradients
evals_f = []
tf = time.time()
for qi in q_list_f:
    q['g'] = qi
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    sorted_evals = np.sort(solver.eigenvalues.real)
    evals_f.append(sorted_evals[:10])

tf = time.time() - tf

logger.info('Time taken for coarse solve = %f' % (tc))
logger.info('Time taken for fine solve = %f' % (tf))

logger.info('Speedup = %f' % (tf/tc))

evals_f = np.array(evals_f)
evals   = np.array(evals)
grads   = np.array(grads)
# Plot
fig = plt.figure(figsize=(6, 4))

plt.plot(q_list_f, evals_f[:, 0::2], '.', c='C0')
plt.plot(q_list_f, evals_f[:, 1::2], '.', c='C1')

# Interpolate between points with cubic splines
for j in range(10):
    for i in range(len(q_list) - 1):
        # Interpolate
        q_fine = np.linspace(q_list[i],q_list[i+1],10)
        # Fit a cubic
        A,B,C,D = cubeFit(q_list[i],q_list[i+1],evals[i,j],evals[i+1,j],grads[i,j],grads[i+1,j])
        # Interpolate onto a finer grid
        evals_fine = A*q_fine**3 + B*q_fine**2 + C*q_fine + D
        if(np.mod(j,2)==0):
            plt.plot(q_fine,evals_fine, '-', c='C3')
        else:
            plt.plot(q_fine,evals_fine, '-', c='C3')

plt.xlim(q_list.min(), q_list.max())
plt.ylim(-10, 30)
plt.xlabel("q")
plt.ylabel("eigenvalues")
plt.title("Mathieu eigenvalues with interpolation")
plt.tight_layout()
plt.savefig("mathieu_eigenvalues_interpolation.png", dpi=200)