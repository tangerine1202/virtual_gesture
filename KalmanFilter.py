from FilterABC import PosVelFilter
from utils import block_diagonal_array
import numpy as np
# various filter: https://filterpy.readthedocs.io/en/latest/
from filterpy.gh import GHFilter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanFilter(PosVelFilter):
    def __init__(self):
        pass

    def build(self, z=None):
        pass

    def predict(self):
        pass

    def update(self, z):
        pass

    def save(self):
        pass

    def reset(self):
        pass

    @property
    def x(self):
        pass

    @property
    def dx(self):
        pass


def pos_vel_filter(x, P, R, Q=0., dt=1.):
    """
    Returns a Kalman Filter which implements a
    constant velocity model for a state [x dx].T
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]])  # position and velocity
    kf.F = np.array([[1., dt],     # state transition matrix
                    [0., 1.]])
    kf.H = np.array([[1., 0]])    # measurement function

    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                   # covariance matrix
    else:
        kf.P[:] = P                 # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q *= Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q

    return kf


def multi_pos_vel_filter(x, P, R, Q=0., dt=1.):
    """
    Returns a Kalman Filter which implements a
    constant velocity model for # of dim states.

    Parameters
    ---
    x: np.array (x0, dx0, x1, dx1, ..., n, dxn)
    """
    dim = len(x) // 2
    kf = KalmanFilter(dim_x=dim*2, dim_z=dim)
    # position and velocity
    kf.x = np.array(x)
    # state transition matrix
    kf.F = block_diagonal_array(dim, [[1., dt], [0., 1.]])
    kf.H = block_diagonal_array(dim, [[1, 0]])    # measurement function

    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                   # covariance matrix
    else:
        kf.P[:] = P                 # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q *= Q_discrete_white_noise(dim=2, dt=dt, var=Q, block_size=dim)
    else:
        kf.Q[:] = Q

    return kf
