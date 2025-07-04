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
import logging, time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from scipy.interpolate import CubicHermiteSpline
logger = logging.getLogger(__name__)

# Parameters
N = 32
q_list = np.linspace(0, 30, 7)
q_list_f = np.linspace(0, 30, 50)
# Basis
coord = d3.Coordinate('x')
dist = d3.Distributor(coord, dtype=np.complex128)
basis = d3.ComplexFourier(coord, N, bounds=(0, 2*np.pi))

# Fields
y = dist.Field(name='y',bases=basis)
a = dist.Field(name='a')

# Substitutions
x = dist.local_grid(basis)
q = dist.Field(name='q')
cos_2x = dist.Field(name='cos_2x', bases=basis)
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
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True, left=True)
    sorted_evals = np.sort(solver.eigenvalues.real)
    indices = np.argsort(solver.eigenvalues.real)
    # Use the left eigenvectors to calculate the gradients
    sub_grad = []
    for index in range(10):
        # Compute gradient of eigenvalues with respect to q
        grad = solver.compute_sensitivity(q, indices[index], solver.subsystems[0])
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
plt.plot(q_list_f, evals_f[:, 0::2], '.', color='C0')
plt.plot(q_list_f, evals_f[:, 1::2], '.', color='C1')

for i in range(10):
    # Fit a cubic Hermite spline
    spline = CubicHermiteSpline(q_list, evals[:, i], grads[:, i])
    # Interpolate onto a finer grid
    evals_fine = q_list_f
    if np.mod(i, 2) == 0:
        plt.plot(q_list_f, spline(q_list_f), '-', color='C0')
    else:
        plt.plot(q_list_f, spline(q_list_f), '-', color='C1')

plt.xlim(q_list.min(), q_list.max())
plt.ylim(-10, 30)
plt.xlabel("q")
plt.ylabel("eigenvalues")
plt.title("Mathieu eigenvalues with interpolation")
plt.tight_layout()
plt.savefig("mathieu_eigenvalues_interpolation.pdf")
