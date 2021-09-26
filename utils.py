import numpy as np
from scipy.spatial.transform import Rotation


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def block_diagonal_array(n, block):
    """
    Returns block diagonal array
    n: # of block
    block: block array
    """
    block = np.array(block)
    arr = None
    for i in range(n):
        tmp = np.array(block)
        tmp = np.hstack([
            np.zeros((block.shape[0], block.shape[1]*i)),
            tmp,
            np.zeros((block.shape[0], block.shape[1]*(n-1-i))),
        ])

        if arr is None:
            arr = tmp
        else:
            arr = np.vstack([arr, tmp])
    return arr


def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # angle in radian
    angle = np.arccos(dot_product)
    return angle


def orthogonal_projection(v1, v2):
    """
    Project vector v1 onto vector v2
    @return orthogonal_vector , orthogonal_complement
    https://ccjou.wordpress.com/2010/04/19/%E6%AD%A3%E4%BA%A4%E6%8A%95%E5%BD%B1-%E5%A8%81%E5%8A%9B%E5%BC%B7%E5%A4%A7%E7%9A%84%E4%BB%A3%E6%95%B8%E5%B7%A5%E5%85%B7/
    """
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1

    z = np.array(v1).reshape(-1, 1)
    x = np.array(v2).reshape(-1, 1)
    P = x @ x.T / (x.T @ x)
    L = np.eye(x.shape[0]) - P
    return L @ z, P @ z


def get_coordinate_rotation_matrix(new_axis, old_axis):
    """
    Calculate rotation matrix for rotating from old-axis to new_axis
    new_axis: [new_axis_1, new_axis_2]
    old_axis: [old_axis_1, old_axis_2]
    note: inverse of rotation matrix = transpose of rotation matrix
    @return rotation matrix
    """
    new_axis = np.asarray(new_axis)
    old_axis = np.asarray(old_axis)
    assert new_axis.shape == (2, 3)
    assert old_axis.shape == (2, 3)
    assert np.isclose(np.dot(new_axis[0], new_axis[1]), 0)
    assert np.isclose(np.dot(old_axis[0], old_axis[1]), 0)

    new_x = new_axis[0] / np.linalg.norm(new_axis[0])
    new_y = new_axis[1] / np.linalg.norm(new_axis[1])
    old_x = old_axis[0] / np.linalg.norm(old_axis[0])
    old_y = old_axis[1] / np.linalg.norm(old_axis[1])

    # rotate x-axis
    cross_x = np.cross(old_x, new_x)
    cross_x = cross_x / np.linalg.norm(cross_x)
    theta_x = angle_between_vectors(old_x, new_x)

    r_x = Rotation.from_rotvec(cross_x * theta_x)

    old_y2 = old_y @ r_x.as_matrix()

    cross_y = np.cross(old_y2, new_y)
    cross_y = cross_y / np.linalg.norm(cross_y)
    theta_y = angle_between_vectors(old_y2, new_y)

    r_y = Rotation.from_rotvec(cross_y * theta_y)

    r = r_x.as_matrix() @ r_y.as_matrix()

    return r
