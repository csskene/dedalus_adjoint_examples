# FitzHugh-Nagumo phase sensitivity analysis

This folder contains code for finding the phase sensitivity function for the FitzHugh-Nagumo equation.

In order to find the phase function, there is an optional *Julia* script. To run this, first ensure *Julia* is installed. This can be done by following the instructions [here](https://julialang.org/install/). A *Julia* environment for running the script can then be created by running
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Once the environment has been setup, the *Julia* script can be run as
```bash
julia --project=. phase_func.jl
```
To run the code multithreaded use
```bash
julia -t <num_threads> --project=. phase_func.jl
```
where `<num_threads>` is user specified.
