# Dedalus Adjoint Examples

This is an examples repository for problems utilizing automatic discrete adjoints in Dedalus.
The repository includes the examples described in [1] as well as several others.
They can be ran using the `adjoint` branch from [the main Dedalus repository](https://github.com/DedalusProject/dedalus).

The extra requirements needed to run the examples can be installed with
```
pip3 install -r requirements.txt
```

## Installing with conda

To build a new conda environment for running the adjoint code, follow these steps:

```
# Setup new conda environment
conda create -n dedalus_adjoint
conda activate dedalus_adjoint

# Macs with Apple Silicon only -- uncomment this to run with x86 instead of arm64
# conda config --env --set subdir osx-64

# Install Dedalus plus build tools
conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1
conda install -c conda-forge dedalus c-compiler cython setuptools wheel

# Replace Dedalus with source installation from adjoint branch
conda uninstall --force dedalus
CC=mpicc pip install --no-cache --no-build-isolation git+https://github.com/dedalusproject/dedalus@adjoint

# Install extra adjoint requirements (from the clone of dedalus_adjoint_examples)
pip install -r requirements.txt
```

## References

[1] [Skene & Burns, "Fast automated adjoints for spectral PDE solvers"](https://arxiv.org/abs/2506.14792)
