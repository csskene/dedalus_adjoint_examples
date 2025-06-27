'''
Helper functions for adjoint looping
'''
import numpy as np
from mpi4py import MPI
import tempfile
from checkpoint_schedules import StorageType, Forward, Reverse, Copy, Move, EndForward, EndReverse
import copy
import uuid
import os
import logging
from dedalus.extras.flow_tools import GlobalArrayReducer
import functools, sys
from scipy.stats import linregress

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
logger = logging.getLogger(__name__)
reducer = GlobalArrayReducer(MPI.COMM_WORLD)

def Taylor_test(cost, grad, point_0, point_p, initial_eps=1e-4):
    '''
    Performs the Taylor test to verify the gradient
    '''
    logger.info('Performing Taylor test')
    if isinstance(point_0, list):
        # We are on a product manifold
        manifold_product_len = len(point_0)
        product_manifold = True
    else:
        manifold_product_len = 1
        product_manifold = False
    residual = []
    if product_manifold:
        cost_0 = cost(*point_0)
        grad_0 = grad(*point_0)
        dJ = 0
        for i in range(manifold_product_len):
            dJ += np.vdot(grad_0[i], point_p[i])
    else:
        cost_0 = cost(point_0)
        grad_0 = grad(point_0)
        dJ = np.vdot(grad_0, point_p)
    eps = initial_eps
    eps_list = []
    for i in range(10):
        eps_list.append(eps)
        if product_manifold:
            point = [point_0[j] + eps*point_p[j] for j in range(manifold_product_len)]
            cost_p = cost(*point)
        else:
            point = point_0 + eps*point_p
            cost_p = cost(point)
        residual.append(np.abs(cost_p - cost_0 - eps*dJ))
        eps /= 2
    regression = linregress(np.log(eps_list), y=np.log(residual))
    return regression.slope, eps_list, residual


# TEMP for now
class global_to_local:
    """
    Helper routines to convert between global and local representations
    of vectors
    """
    def __init__(self, weight_layout, field):
        self.weight_layout = weight_layout
        self.tensor_shape = tuple(cs.dim for cs in field.tensorsig)
        self.global_shape = self.tensor_shape + self.weight_layout.global_shape(field.domain, 1)
        self.local_slice = tuple(slice(None) for cs in field.tensorsig) + self.weight_layout.slices(field.domain, 1)

    # Convert between vector and fields
    def vector_to_field(self, vector, field):
        field.change_scales(1)
        vec = vector.reshape(self.global_shape)
        field[self.weight_layout] = vec[self.local_slice]

    def field_to_vector(self, vector, field):
        field.change_scales(1)
        vector[:] = field.allgather_data(layout=self.weight_layout).flatten()

class direct_adjoint_loop:
    """
    Class to perform a direct adjoint loop in Dedalus

    Parameters
    ----------
    solver : solver
            A Dedalus IVP solver
    max_iterations: int 
                  The stop iteration for the IVP
    timestep: float
             The constant timestep to use
    cost functional: Dedalus expression
                    A Dedalus expression for the cost functional
    pre_solvers: list
                A list of linear Dedalus solvers to be excecuted
                before solving the IVP. 
    post_solvers: list
                A list of linaer Dedalus solvers to be excecuted
                after solving the IVP
    """
    def __init__(self, solver, max_iterations, timestep, cost_functional, adjoint_dependencies=[], pre_solvers=[], post_solvers=[]):
        self.state = solver.state
        self.adjoint_dependencies = adjoint_dependencies
        self.solver = solver
        self.solver.stop_iteration = max_iterations
        self.timestep = timestep
        self.J = cost_functional
        self.pre_solvers = pre_solvers
        self.post_solvers = post_solvers
        # solve pre_solvers now to set IC for IVP
        for pre_solver in self.pre_solvers:
            pre_solver.solve()
        # Work memory for checkpointing
        self.forward_work_memory = {StorageType.WORK: {}}
        self.forward_work_memory[StorageType.WORK][0] = copy.deepcopy([field['c'] for field in self.state])
        self.forward_final_solution = None
        self.initial_condition = copy.deepcopy([field['c'] for field in self.state])
        self.adjoint_work_memory = {StorageType.WORK: {}}
        self.restart_forward = {StorageType.RAM: {}, StorageType.DISK: {}}
        self.adjoint_dependency = {StorageType.WORK: {}, StorageType.RAM: {}, StorageType.DISK: {}}
        self.mode = "forward"

    def reset_initial_condition(self):
        """
        Resets the IC to the current values of solver.state
        """
        self.mode = "forward"
        for pre_solver in self.pre_solvers:
            pre_solver.solve()
        self.forward_work_memory[StorageType.WORK][0] = copy.deepcopy([field['c'] for field in self.state])

    # Interface for checkpoint_schedules
    def forward(self, n0, n1, storage=None, write_adj_deps=False, write_ics=False):
        """Advance the direct solver from n0 to n1

        Parameters
        ---------
        n0 : int
            Initial time step.
        n1 : int
            Final time step.
        write_adj_deps : bool
            If `True`, the adjoint dependency data will be saved.
        write_ics : bool
            If `True`, the IC dependency data will be saved.
        """
        # Reset RHS evaluators to correct iteration
        self.solver.reset(iter=n0)
        initial_time_state = self.forward_work_memory[StorageType.WORK].pop(n0)
        for i, field in enumerate(self.state):
            field['c'] = initial_time_state[i]
        if write_ics:
            self._store_data([field['c'] for field in self.state], n0, storage, write_adj_deps, write_ics)
        # Evolve from n0 to n1
        for step in range(n0, min(n1, self.solver.stop_iteration)):
            if write_adj_deps:
                self._store_data([field['c'] for field in self.adjoint_dependencies], step, storage, write_adj_deps, write_ics)
            self.solver.step(self.timestep)
        step += 1
        if step == self.solver.stop_iteration:
            for post_solver in self.post_solvers:
                post_solver.solve()
            self.forward_final_solution = copy.deepcopy([field['c'] for field in self.state])
            # Comment as IVP does not need the last one
            # if write_adj_deps:
            #     self.adjoint_dependency[StorageType.WORK][step] = copy.deepcopy(self.state[0]['c'])
        if (not write_adj_deps
           or (self.mode == "forward" and step < (self.solver.stop_iteration))
        ):
            self.forward_work_memory[StorageType.WORK][step] = copy.deepcopy([field['c'] for field in self.state])

    def adjoint(self, n0, n1, clear_adj_deps):
        """Advance the adjoint solver from n1 to n0

        Parameters
        ---------
        n0 : int
            Initial time step.
        n1 : int
            Final time step.
        clear_adj_deps : bool
            If `True`, the adjoint dependency data will be cleared.
        """
        self.mode = "adjoint"
        if n1 == self.solver.stop_iteration:
            self._initialize_adjoint()
        final_time_state = self.adjoint_work_memory[StorageType.WORK].pop(n1)
        for i, field in enumerate(self.adjoint_state):
            field['c'] = final_time_state[i]
        for step in range(n1, n0, - 1):
            for i, field in enumerate(self.adjoint_dependencies):
                field['c'] = self.adjoint_dependency[StorageType.WORK][step-1][i]
            self.solver.step_adjoint(self.timestep)
            if clear_adj_deps:
                del self.adjoint_dependency[StorageType.WORK][step-1]
        step -= 1
        self.adjoint_work_memory[StorageType.WORK][step] = copy.deepcopy([field['c'] for field in self.adjoint_state])

    def gradient(self):
        """Compute the adjoint-based gradient.

        Returns
        -------
        array
            The adjoint-based gradient.
        """
        for pre_solver in reversed(self.pre_solvers):
            self.cotangents = pre_solver.compute_sensitivities(self.cotangents)
        return self.cotangents

    def functional(self):
        """Compute the cost functional.

        Returns
        -------
        scalar
            The cost functional
        """
        for i, field in enumerate(self.state):
            field['c'] = self.forward_final_solution[i]
        return reducer.global_max(self.J['g'])

    def copy_data(self, step, from_storage, to_storage, move=False):
        """Copy data from one storage to another.

        Parameters
        ----------
        step : int
            The time step.
        from_storage : StorageType
            The storage type from which the data will be copied.
        to_storage : StorageType
            The storage type to which the data will be copied.
        move : bool, optional
            Whether the data will be moved or not. If `True`, the data will be
            removed from the `from_storage`.
        """
        if from_storage == StorageType.DISK:
            if step in self.adjoint_dependency[StorageType.DISK]:
                file_name = self.adjoint_dependency[StorageType.DISK][step]
                with np.load(file_name) as data:
                    state_data = []
                    for i in range(len(data.keys())):
                        state_data.append(data[str(i)])
                    self.adjoint_dependency[to_storage][step] = state_data
            if step in self.restart_forward[StorageType.DISK]:
                file_name = self.restart_forward[StorageType.DISK][step]
                with np.load(file_name) as data:
                    state_data = []
                    for i in range(len(data.keys())):
                        state_data.append(data[str(i)])
                    self.forward_work_memory[to_storage][step] = state_data 
            if move:
                os.remove(file_name)
        elif from_storage == StorageType.RAM:
            self.forward_work_memory[to_storage][step] = \
                copy.deepcopy(self.restart_forward[from_storage][step])
            if move:
                if step in self.adjoint_dependency[from_storage]:
                    del self.adjoint_dependency[from_storage][step]
                if step in self.restart_forward[from_storage]:
                    del self.restart_forward[from_storage][step]
        else:
            raise ValueError("This `StorageType` is not supported.")
        
    def _initialize_adjoint(self):
        # For adjoint
        self.cotangents = {}
        Jadj = self.J.evaluate().copy_adjoint()
        Jadj['g'] = 1
        self.cotangents[self.J] = Jadj
        id = uuid.uuid4()
        _, self.cotangents =  self.J.evaluate_vjp(self.cotangents, id=id, force=True)
        for post_solver in reversed(self.post_solvers):
            self.cotangents = post_solver.compute_sensitivities(self.cotangents)
        self.solver.state_adj = []
        self.adjoint_state = self.solver.state_adj
        for state in self.state:
            if state in self.cotangents:
                self.adjoint_state.append(self.cotangents[state])
            else:
                adjoint_state = state.copy_adjoint()
                adjoint_state.preset_layout('c')
                adjoint_state.data *= 0
                self.adjoint_state.append(adjoint_state)
                self.cotangents[state] = adjoint_state
        self.adjoint_work_memory[StorageType.WORK][self.solver.stop_iteration] = copy.deepcopy([field['c'] for field in self.adjoint_state])

    def _store_data(self, data, step, storage, write_adj_deps, write_ics):
        if storage == StorageType.DISK:
            self._store_on_disk(data, step, write_adj_deps)
        elif write_adj_deps:
            self.adjoint_dependency[storage][step] = copy.deepcopy(data)
        elif write_ics:
            self.restart_forward[storage][step] = copy.deepcopy(data)

    def _store_on_disk(self, data, step, adj_deps):
        if adj_deps:
            tmp_adj_deps_directory = tempfile.gettempdir()
            file_name = os.path.join(tmp_adj_deps_directory, "fwd_" + str(step) \
                                     + "_rank_" + str(rank) + ".npz")
            self.adjoint_dependency[StorageType.DISK][step] = file_name
            save_dict = {}
            for idx, variable in enumerate(data):
                save_dict[str(idx)] = variable
            np.savez(file_name, **save_dict)
        else:
            tmp_rest_forward_directory = tempfile.gettempdir()
            file_name = os.path.join(tmp_rest_forward_directory, "fwd_" + str(step) \
                                     + "_rank_" + str(rank) + ".npz")
            self.restart_forward[StorageType.DISK][step] = file_name
            save_dict = {}
            for idx, variable in enumerate(data):
                save_dict[str(idx)] = variable
            np.savez(file_name, **save_dict)

    # def _clean_disk(self):
    #     if len(self.adjoint_dependency[StorageType.DISK]) > 0:
    #         for step in self.adjoint_dependency[StorageType.DISK]:
    #             print(self.adjoint_dependency[StorageType.DISK][step])
    #             # self.adjoint_dependency[StorageType.DISK][step].close()
    #     if len(self.restart_forward[StorageType.DISK]) > 0:
    #         for step in self.restart_forward[StorageType.DISK]:
    #             print(self.restart_forward[StorageType.DISK][step])
    #             # self.restart_forward[StorageType.DISK][step].close()

class CheckpointingManager:
    """Manage the forward and adjoint solvers.

    Attributes
    ----------
    schedule : CheckpointSchedule
        The schedule created by `checkpoint_schedules` package.
    solver : object
        A solver object used to solve the forward and adjoint solvers.
    
    Notes
    -----
    The `solver` object contains methods to execute the forward and adjoint. In 
    addition, it contains methods to copy data from one storage to another, and
    to set the initial condition for the adjoint.
    """
    def __init__(self, create_schedule, solver):
        self.solver = solver
        self.create_schedule = create_schedule
        
    def execute(self, mode='forward'):
        """Execute forward/adjoint using checkpointing.
        """
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")

        @action.register(Forward)
        def action_forward(cp_action):
            n1 = cp_action.n1
            self.solver.forward(cp_action.n0, n1, storage=cp_action.storage,
                                  write_adj_deps=cp_action.write_adj_deps,
                                  write_ics=cp_action.write_ics)
            if n1 >= self.solver.solver.stop_iteration:
                n1 = min(n1, self.solver.solver.stop_iteration)
                self._schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            self.solver.adjoint(cp_action.n0, cp_action.n1, cp_action.clear_adj_deps)
            self.reverse_step += cp_action.n1 - cp_action.n0
            
        @action.register(Copy)
        def action_copy(cp_action):
            self.solver.copy_data(cp_action.n, cp_action.from_storage,
                                    cp_action.to_storage, move=False)

        @action.register(Move)
        def action_move(cp_action):
            self.solver.copy_data(cp_action.n, cp_action.from_storage,
                                    cp_action.to_storage, move=True)
            
        @action.register(EndForward)
        def action_end_forward(cp_action):
            if self._schedule.max_n is None:
                self._schedule._max_n = self.max_n
            
        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            # self.solver._clean_disk()
            if self._schedule.max_n != self.reverse_step:
                raise ValueError("The number of steps in the reverse phase"
                                 "is different from the number of steps in the"
                                 "forward phase.")
        if mode=='forward' or mode=='both':
            self.max_n = sys.maxsize
            self._schedule = self.create_schedule()
        self.reverse_step = 0
        for _, cp_action in enumerate(self._schedule):
            action(cp_action)
            if isinstance(cp_action, EndForward) and mode=='forward':
                break
            elif isinstance(cp_action, EndReverse) and (mode=='reverse' or mode=='both'):
                break
        