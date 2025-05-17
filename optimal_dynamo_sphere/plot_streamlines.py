"""
Plot streamlines from VTK data (run convert_to_vtk first!)

Usage:
    plot_streamlines.py 
"""
import pyvista as pv

task = 'u'
mesh = pv.read('snapshot_VTK.vts')
mesh.set_active_scalars(task)

p = pv.Plotter(window_size=(1000, 1000))
# p.show_axes()

p.add_mesh(pv.Sphere(theta_resolution=64, phi_resolution=64, radius=1), color='gray', opacity=0.05)

streamlines = mesh.streamlines(source_radius=0.1, n_points=75, terminal_speed=1e-3, max_step_length=0.01, max_length=1.5, integration_direction='both')
p.add_mesh(streamlines.tube(radius=0.01), scalars=task, lighting=True, clim=(-0.1, 0.6))
# Adjust camera position
scale = 0.9
p.camera_position = [(-3.048601987406693*scale, 2.890153121510564*scale, -1.7990926290671938*scale),
(0.0, 0.0, 0.0),
(0.7268850704018675, 0.43689992323199267, -0.5298646539511305)]
p.remove_scalar_bar()
p.show(auto_close=False)
print(p.camera_position)
p.screenshot('streamlines.png', scale=2)
