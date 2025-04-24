# Calculates the phase function
# (Run phase-reduction.py first!)
using NPZ
using DifferentialEquations
using LinearAlgebra
using HDF5

println("Number of threads: ", Threads.nthreads())

# Read in limit cycle
file = h5open("FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5", "r")
u0 = read(file["tasks/u0"])[:, 1]
v0 = read(file["tasks/v0"])[:, 1]
t = range(0, 2pi, length=length(u0)+1)
t = t[1:end-1]
close(file)

LC = hcat(u0, v0)
T = 36.518032

# Define the ODE
function fitzhugh_nagumo!(du, q, p, t)
    u, v = q
    a, b, eps, I = p
    du[1] = u - u^3/3 - v + I
    du[2] = eps*(u + a - b*v)
end

# Set up an esemble problem
q0 = [0.0, 0.0]
p = [0.7, 0.8, 0.08, 0.8]
tspan = (0.0, 3T)

U = range(-2.2, 2.2, length=400)
V = range(-0.2, 1.8, length=200)
grid = [[u, v] for u in U, v in V]

prob = ODEProblem(fitzhugh_nagumo!, q0, tspan, p)

function prob_func(prob, i, repeat)
    remake(prob, u0=grid[i])
end

function output_func(sol, i)
    # Output the phase
    # Assign the phase to that of the closest 
    # point on the LC that the final solution 
    # approaches
    final_state = sol.u[end]
    distances = zeros(size(LC, 1))
    for j=1:size(LC, 1)
        distances[j] = norm(final_state - LC[j, :])
    end
    closest_index = argmin(distances)
    phase = t[closest_index]
    phase, false
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)

# Solve the ensemble problem
@time begin
    sol = solve(ensemble_prob, RK4(), EnsembleThreads(), trajectories=length(grid))
end
# Write the phase function in npz format
npzwrite("phase_func.npz", Dict("phase_func" => reshape(sol.u, length(U), length(V)), "u" => U, "v" => V))
 