"""
Plot streamlines from VTK data (run convert_to_vtk first!)

Usage:
    plot_3D.py <files>... [--output=<dir>]
"""
import pyvista as pv

task = 'B'
mesh = pv.read('snapshot_VTK.vts')
mesh.set_active_scalars(task)

p = pv.Plotter()
p.show_axes()

p.add_mesh(pv.Sphere(theta_resolution=64, phi_resolution=64, radius=1), color='gray', opacity=0.05)

streamlines = mesh.streamlines(vectors=task, source_radius=0.1, n_points=75, terminal_speed=1e-3, max_step_length=0.01, max_length=10, integration_direction='both', max_steps=100000)
p.add_mesh(streamlines.tube(radius=0.01), scalars=task, lighting=True)

p.show(auto_close=False)
p.screenshot('streamlines.png')
