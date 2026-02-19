# Calculates the phase function
# (Run phase_reduction.py first!)
using NPZ
using DifferentialEquations
using LinearAlgebra
using HDF5
using NearestNeighbors

println("Number of threads: ", Threads.nthreads())

# Parameters
Nu = 400
Nv = 200
num_trajectories = Nu*Nv
T = 36.518032
p = [0.7, 0.8, 0.08, 0.8]
tspan = (0.0, 3T)
h5path = "FitzHugh-Nagumo/FitzHugh-Nagumo_s1.h5"

# Read in limit cycle
isfile(h5path) || error("Required HDF5 file not found: $(abspath(h5path)). Run phase_reduction.py first.")

file = h5open(h5path, "r")
u0 = real.(read(file["tasks/u0"])[:, 1])
v0 = real.(read(file["tasks/v0"])[:, 1])
t = range(0, 2π, length=length(u0)+1)
t = t[1:end-1]
close(file)

LC = hcat(u0, v0)

# Define the ODE
function fitzhugh_nagumo!(du, q, p, t)
    u, v = q
    a, b, eps, I = p
    du[1] = u - u^3/3 - v + I
    du[2] = eps*(u + a - b*v)
end

# Set up an ensemble problem
q0 = zeros(2)
U = range(-2.2, 2.2, length=Nu)
V = range(-0.2, 1.8, length=Nv)

prob = ODEProblem(fitzhugh_nagumo!, q0, tspan, p)

@inline function idx_to_uv(i, U, V, Nu)
    iu = (i - 1) % Nu + 1 # index into U
    iv = (i - 1) ÷ Nu + 1 # index into V
    return U[iu], V[iv]
end

function prob_func(prob, i, repeat)
    u, v = idx_to_uv(i, U, V, Nu)
    remake(prob; u0=[u, v])
end

const kdtree = KDTree(permutedims(LC))
function output_func(sol, i)
    uT, vT = sol.u[end]
    idxs, _ = knn(kdtree, [uT, vT], 1, true)
    return t[idxs[1]], false
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)

# Solve the ensemble problem
@time begin
    sol = solve(ensemble_prob, RK4(), EnsembleThreads(), save_start=false, 
                save_everystep=false, trajectories=num_trajectories)
end
# Write the phase function in npz format
npzwrite("phase_func.npz", Dict("phase_func" => reshape(sol.u, length(U), length(V)), "u" => U, "v" => V))
 