"""
Phase reduction analysis for the FitzHugh-Nagumo model
which describes the firing dynamics of a Neuron [1]

The script discretises the limit-cycle with period T using
a Fourier basis. This limit cycle is converged using Newton
iteration.

The phase sensitivity function is then found by computing the 
adjoint Floquet mode corresponding to the neutral direct
Floquet mode which describes phase shifts along the limit
cycle.

This script should only take a few seconds to run (serial only)

Useage:
    python3 phase_reduction.py

[1] Phase reduction approach to synchronisation of nonlinear oscillators, 
    Hiroya Nakao, Contemporay Physics, 57:2, 188-214
"""
import logging
import numpy as np
import dedalus.public as d3
import dedalus.core.evaluator as evaluator

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

if __name__=="__main__":
    # Parameters
    N = 512  # Resolution
    a = 0.7
    b = 0.8
    eps = 0.08
    I = 0.8
    u_guess = 1.06255285
    v_guess = 1.58174182
    T_guess = 36.5181189 

    # Coordinates and bases
    coords = d3.CartesianCoordinates('t')
    dist = d3.Distributor(coords, dtype=np.complex128)
    tbasis = d3.ComplexFourier(coords['t'], size=N, bounds=(0, 2*np.pi), dealias=2)
    t = dist.local_grid(tbasis)

    # Fields
    # Limit cycle fields
    u0 = dist.Field(name='u0', bases=tbasis)
    v0 = dist.Field(name='v0', bases=tbasis)
    T0 = dist.Field(name='T0')
    # Perturbation fields
    u = dist.Field(name='u', bases=tbasis)
    v = dist.Field(name='v', bases=tbasis)
    T = dist.Field(name='T')
    # Phase sensitivity
    Zu = dist.Field(name='Zu', bases=tbasis, adjoint=True)
    Zv = dist.Field(name='Zv', bases=tbasis, adjoint=True)
    
    # Generate nice initial guess for the
    # entire limit cycle from ODE
    ug = dist.Field(name='ug')
    vg = dist.Field(name='vg')

    problem = d3.IVP([ug, vg], namespace=locals())
    problem.add_equation("dt(ug) = ug - ug**3/3-vg+I")
    problem.add_equation("dt(vg) = eps*(ug + a - b*vg)")
    solver = problem.build_solver(d3.RK443)

    ug['g'] = u_guess
    vg['g'] = v_guess

    u_list = []
    v_list = []
    timestep = T_guess/N
    solver.stop_sim_time = T_guess
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        u_list.append(np.max(ug['g']))
        v_list.append(np.max(ug['g']))

    # Converge limit cycle using Newton iteration
    # Write manually so can solve for the limit cycle period

    dt = lambda A: d3.Differentiate(A, coords['t']) # Now safe to change dt
    du0dt = dt(u0) # Time derivative of the limit cycle - u
    dv0dt = dt(v0) # Time derivative of the limit cycle - v

    RHS_u = 2*np.pi*dt(u0)/T0 - u0 + u0**3/3 + v0 - I 
    RHS_v = 2*np.pi*dt(v0)/T0 - eps*(u0 + a - b*v0)

    u0['g'] = u_list[:N]
    v0['g'] = v_list[:N]
    T0['g'] = T_guess

    problem = d3.LBVP([u, v, T], namespace=locals())
    problem.add_equation("2*np.pi*dt(u)/T0 - 2*np.pi*dt(u0)/T0**2*T -u + u0**2*u + v = -RHS_u")
    problem.add_equation("2*np.pi*dt(v)/T0 - 2*np.pi*dt(v0)/T0**2*T - eps*(u - b*v) = -RHS_v")
    problem.add_equation("integ(u*du0dt) + integ(v*dv0dt) = 0") # Dont shift phase!
    solver = problem.build_solver()

    pert_norm = np.inf
    while pert_norm>1e-12:
        solver.solve(rebuild_matrices=True)
        u0['c'] += u['c']
        v0['c'] += v['c']
        T0['c'] += T['c']
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in [u, v, T])
        logger.info(f'Perturbation norm: {pert_norm:.3e}')

    logger.info('T = {0:f}'.format(np.max(T0['g'])))
    # Fix phase such that u0['c'] is imaginary and negative
    c0 = u0['c'][1]
    phi = np.arctan(c0.real/c0.imag)
    if c0*np.exp(1j*phi)>0:
        phi += np.pi
    u0['c'] *= np.exp(1j*phi*tbasis.wavenumbers)
    v0['c'] *= np.exp(1j*phi*tbasis.wavenumbers)

    # Save LC for calculating phase function
    u0.change_scales(1)
    v0.change_scales(1)
    np.savez('phase_npy', u=u0['g'], v=v0['g'], t=t) 

    # Solve the Floquet problem
    omega = dist.Field(name='omega')
    problem = d3.EigenvalueProblem([u, v], eigenvalue=omega, namespace=locals())
    problem.add_equation("2*np.pi*omega*u/T0 + 2*np.pi*dt(u)/T0 - u + u0**2*u + v= 0")
    problem.add_equation("2*np.pi*omega*v/T0 + 2*np.pi*dt(v)/T0 - eps*u + eps*b*v = 0")
    solver = problem.build_solver(ncc_cutoff=1e-12)
    nev = 1  #Find one eigenvalue
    target = 0  #Target the phase shift with eigenvalue 0
    solver.solve_sparse(solver.subproblems[0], nev, target, left=True)
    logger.info('Floquet mode found has eigenvalue %g+1j(%g)' % (solver.eigenvalues[0].real, solver.eigenvalues[0].imag))

    # Normalise the phase sensitivity function
    set_state_adjoint(solver, 0, solver.subsystems[0])
    Zu['c'] = u['c']
    Zv['c'] = v['c']
    norm_field = np.conj(Zu)*du0dt + np.conj(Zv)*dv0dt # This function should be constant
    norm = np.mean((np.conj(Zu)*du0dt + np.conj(Zv)*dv0dt)['g'])
    Zu.change_scales(1)
    Zv.change_scales(1)
    Zu['g'] /= np.conj(norm)
    Zv['g'] /= np.conj(norm)
    norm = np.mean((np.conj(Zu)*du0dt + np.conj(Zv)*dv0dt)['g'])
    logger.info("This should be 1: %f+1j*(%f)" % (norm.real, norm.imag))
    solver.set_state(0, solver.subsystems[0])
    # Normalise with the phase-shift given by du0/dt, dv0/dt
    u['c'] /= u['c'][1]/du0dt['c'][1]
    v['c'] /= v['c'][1]/dv0dt['c'][1]

    # Save output
    output_evaluator = evaluator.Evaluator(dist, locals())
    output_handler = output_evaluator.add_file_handler('FitzHugh-Nagumo')
    output_handler.add_tasks([u0, v0, T0, du0dt, dv0dt, u, v, Zu, Zv, norm_field])
    output_handler.add_task(du0dt, name='du0dt')
    output_handler.add_task(dv0dt, name='dv0dt')
    output_handler.add_task(norm_field, name='norm')
    output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)