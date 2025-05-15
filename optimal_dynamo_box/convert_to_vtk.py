"""
Convert cartesian Dedalus h5 snapshot to vtk format

Usage:
    convert_to_vtk.py <files>...
"""
import h5py
import numpy as np
from pyevtk.hl import gridToVTK

def extrapolate_coordinates(x, y, z):
    """Extrapolate data to coordinate boundaries."""
    # Repeat first x point
    x_extrap = np.concatenate([x, [1]])
    # Repeat first y point
    y_extrap = np.concatenate([y, [1]])
    # Repeat first z point
    z_extrap = np.concatenate([z, [1]])
    return x_extrap, y_extrap, z_extrap

def extrapolate_data(*data):
    """Extrapolate data to coordinate boundaries."""
    data = np.array(data)
    # Build data
    shape = np.array(data.shape) + np.array([0, 1, 1, 1])
    data_extrap = np.zeros(shape=shape, dtype=data.dtype)
    # Copy interior data
    data_extrap[:, :-1, :-1, :-1] = data
    # Copy last point in x
    data_extrap[:, -1, :, :] = data_extrap[:, 0, :, :]
    # Copy last point in y
    data_extrap[:, :, -1, :] = data_extrap[:, :, 0, :]
    # Copy last point in z
    data_extrap[:, :, :, -1] = data_extrap[:, :, :, 0]
    return data_extrap

def main(filename, index=-1):
    with h5py.File(filename, mode='r') as file:
        # Load data on simulation grid
        dset = file['tasks']['u']
        x_g = dset.dims[2][0][:].ravel()
        y_g = dset.dims[3][0][:].ravel()
        z_g = dset.dims[4][0][:].ravel()

        u_g = file['tasks']['u'][index]
        B_g = file['tasks']['B'][index]
        helicity_g = file['tasks']['helicity'][index]
        ME_density_g = file['tasks']['ME_density'][index]

        # Extrapolate to full box
        x, y, z = extrapolate_coordinates(x_g, y_g, z_g)
        B = extrapolate_data(B_g[0], B_g[1], B_g[2])
        u = extrapolate_data(u_g[0], u_g[1], u_g[2])
        helicity = extrapolate_data(helicity_g).squeeze()
        ME_density = extrapolate_data(ME_density_g).squeeze()
        # Save to VTK
        pointData = {'B': (B[0], B[1], B[2]), 
                     'u': (u[0], u[1], u[2]), 
                     'helicity': (helicity),
                     'ME_density': (ME_density)}
        gridToVTK("snapshot_VTK", x, y, z, pointData=pointData)

if __name__=='__main__':
    from docopt import docopt
    from dedalus.tools import post

    args = docopt(__doc__)
    main(args['<files>'][0])