"""
Convert spherical Dedalus h5 snapshot to vtk format

Usage:
    convert_to_vtk.py <files>...
"""
import h5py
import numpy as np
from pyevtk.hl import gridToVTK

def spherical_to_cartesian(phi, theta, r):
    phi, theta, r = np.meshgrid(phi, theta, r, indexing='ij')
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_to_cartesian_field(phi_cell, theta_cell, field_data):
    field_phi = field_data[0]
    field_theta = field_data[1]
    field_r = field_data[2]
    field_x = field_r*np.sin(theta_cell)*np.cos(phi_cell) + field_theta*np.cos(theta_cell)*np.cos(phi_cell) - field_phi*np.sin(phi_cell)
    field_y = field_r*np.sin(theta_cell)*np.sin(phi_cell) + field_theta*np.cos(theta_cell)*np.sin(phi_cell) + field_phi*np.cos(phi_cell)
    field_z = field_r*np.cos(theta_cell) - field_theta*np.sin(theta_cell)
    field_data_cart = np.empty_like(field_data)
    field_data_cart[0] = field_x
    field_data_cart[1] = field_y
    field_data_cart[2] = field_z
    return field_data_cart

def main(filename, start, count):
    with h5py.File(filename, mode='r') as file:
        dset = file['tasks/u']
        phi = dset.dims[2][0][:].ravel()
        theta = dset.dims[3][0][:].ravel()
        r = dset.dims[4][0][:].ravel()
        # For now
        phi_cell, theta_cell, r_cell = np.meshgrid(phi, theta, r, indexing='ij')
        dset_cart = spherical_to_cartesian_field(phi_cell, theta_cell, dset[-1])
        x, y, z = spherical_to_cartesian(phi, theta, r)
        gridToVTK("U_VTK", x, y, z, pointData= {"U":(dset_cart[0], dset_cart[1], dset_cart[2])})

if __name__=='__main__':
    from docopt import docopt
    from dedalus.tools import post

    args = docopt(__doc__)
    post.visit_writes(args['<files>'], main)