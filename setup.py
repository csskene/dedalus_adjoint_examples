from setuptools import setup

setup(
    name='dedalus_adjoint_examples',
    version='0.1',
    packages=['adjoint_helper_functions'],
    install_requires=[
        'checkpoint_schedules',
        'pymanopt',
        'matplotlib-label-lines',
        'pyvista',
        'pyevtk'
    ],
    author='C. S. Skene and K. J. Burns',
    description='A collection of Dedalus adjoint examples',
)
