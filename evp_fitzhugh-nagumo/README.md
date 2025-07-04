# FitzHugh-Nagumo phase sensitivity analysis

This folder contains code for finding the phase sensitivity function for the FitzHugh-Nagumo equation.

In order to find the phase function, there is an optional *julia* script. To run this, first ensure *julia* is installed. This can be done by following the instructions [here](https://julialang.org/install/). A *julia* environment for running the script can then be created by running
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
Once the environment has been setup, the julia script can be run as
```
julia --project=. phase_func.jl
```