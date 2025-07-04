"""
Plot isosurfaces from VTK data (run convert_to_vtk first!)

Usage:
    plot_isosurface.py
"""
import pyvista as pv

task = 'helicity'
mesh = pv.read('snapshot_VTK.vtr')
mesh.set_active_scalars(task)

p = pv.Plotter()
p.show_axes()

contours = mesh.contour(isosurfaces=(-0.14, 0.14))

p.add_mesh(pv.Box((0, 1, 0, 1, 0, 1)), color='gray', opacity=0.05)
p.add_mesh(contours, cmap='bwr')

p.show(auto_close=False)
p.screenshot('isosurface.png', scale=1)
