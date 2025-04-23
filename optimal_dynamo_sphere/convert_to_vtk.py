"""
Convert spherical Dedalus h5 snapshot to vtk format

Usage:
    convert_to_vtk.py <files>...
"""
import h5py
import numpy as np
from pyevtk.hl import gridToVTK


def extrapolate_coordinates(phi, theta, r):
    """Extrapolate data to coordinate boundaries."""
    # Repeat first phi point
    phi_extrap = np.concatenate([phi, [2*np.pi]])
    # Add theta endpoints
    theta_extrap = np.concatenate([[np.pi], theta, [0]])
    # Add radial endpoints
    r_extrap = np.concatenate([[0], r, [1]])
    return phi_extrap, theta_extrap, r_extrap

def extrapolate_data(phi, theta, r, *data):
    """Extrapolate data to coordinate boundaries."""
    data = np.array(data)
    # Repeat first phi point
    phi_extrap = np.concatenate([phi, [2*np.pi]])
    # Add theta endpoints
    theta_extrap = np.concatenate([[np.pi], theta, [0]])
    # Add radial endpoints
    r_extrap = np.concatenate([[0], r, [1]])
    # Build data
    shape = np.array(data.shape) + np.array([0, 1, 2, 2])
    data_extrap = np.zeros(shape=shape, dtype=data.dtype)
    # Copy interior data
    data_extrap[:, :-1, 1:-1, 1:-1] = data
    # Copy last point in phi
    data_extrap[:, -1, :, :] = data_extrap[:, 0, :, :]
    # Average around south pole
    data_extrap[:, :, 0, :] = np.mean(data_extrap[:, :, 1, :], axis=1)[:, None, :]
    # Average around north pole
    data_extrap[:, :, -1, :] = np.mean(data_extrap[:, :, -2, :], axis=1)[:, None, :]
    # Average around origin
    data_extrap[:, :, :, 0] = np.mean(data_extrap[:, :, :, 1], axis=(1,2))[:, None, None]
    # Extrapolate to outer radius
    data_extrap[:, :, :, -1] = data_extrap[:, :, :, -2] + (r_extrap[-1] - r_extrap[-2]) / (r_extrap[-2] - r_extrap[-3]) * (data_extrap[:, :, :, -2] - data_extrap[:, :, :, -3])
    return data_extrap



def convert_vector_components(vphi, vtheta, vr, phi, theta):
    vx = np.sin(theta)*np.cos(phi)*vr + np.cos(theta)*np.cos(phi)*vtheta - np.sin(phi)*vphi
    vy = np.sin(theta)*np.sin(phi)*vr + np.cos(theta)*np.sin(phi)*vtheta + np.cos(phi)*vphi
    vz = np.cos(theta)*vr - np.sin(theta)*vtheta
    return vx, vy, vz


def main(filename, index=-1):
    with h5py.File(filename, mode='r') as file:

        # Load data
        dset = file['tasks']['u']
        phi = dset.dims[2][0][:].ravel()
        theta = dset.dims[3][0][:].ravel()
        r = dset.dims[4][0][:].ravel()
        u = file['tasks']['u'][index]
        B = file['tasks']['B'][index]

        # Extrapolate coordinates
        phi_extrap, theta_extrap, r_extrap = extrapolate_coordinates(phi, theta, r)
        phi_extrap = phi_extrap[:, None, None]
        theta_extrap = theta_extrap[None, :, None]
        r_extrap = r_extrap[None, None, :]

        # Convert coordinates
        x = r_extrap * np.sin(theta_extrap) * np.cos(phi_extrap)
        y = r_extrap * np.sin(theta_extrap) * np.sin(phi_extrap)
        z = r_extrap * np.cos(theta_extrap) + np.zeros_like(phi_extrap)

        # Convert vector components
        ux, uy, uz = convert_vector_components(u[0], u[1], u[2], phi[:, None, None], theta[None, :, None])
        Bx, By, Bz = convert_vector_components(B[0], B[1], B[2], phi[:, None, None], theta[None, :, None])

        # Extrapolate vector components
        ux, uy, uz = extrapolate_data(phi, theta, r, ux, uy, uz)
        Bx, By, Bz = extrapolate_data(phi, theta, r, Bx, By, Bz)

        # Clean cartesian values to avoid polar artifacts
        eps = 1e-6
        x[np.abs(x) < eps] = 0
        y[np.abs(y) < eps] = 0
        z[np.abs(z) < eps] = 0

        # Save to VTK
        pointData = {'u': (ux, uy, uz), 'B': (Bx, By, Bz)}
        gridToVTK("snapshot_VTK", x, y, z, pointData=pointData)


if __name__=='__main__':
    from docopt import docopt
    from dedalus.tools import post

    args = docopt(__doc__)
    main(args['<files>'][0])