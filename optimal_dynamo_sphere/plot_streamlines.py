"""
Plot streamlines from VTK data (run convert_to_vtk first!)

Usage:
    plot_3D.py <files>... [--output=<dir>]
"""
import pyvista as pv

mesh = pv.read('U_VTK.vts')
mesh.set_active_scalars('U')
streamlines = mesh.streamlines()

p = pv.Plotter()
streamlines = mesh.streamlines(source_radius=0.1, n_points=100, terminal_speed=0.05, initial_step_length=0.01)
p.add_mesh(streamlines.tube(radius=0.01), scalars='U', lighting=False)
p.show(auto_close=False)
p.screenshot('streamlines.png') 
