from lib.math_utils import block_diagonal_array
from filter.FilterABC import PosVelFilter
import numpy as np
# various filter: https://filterpy.readthedocs.io/en/latest/
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver


class KalmanPosVelFilter(PosVelFilter):
    """
    Estimates are arraged in [pos0, vel0, pos1, vel1, ...] order.
    """

    def __init__(self, dim_z, P, R, Q=0., dt=1., should_save=False):
        # position & velocity variables
        self.dim_x = dim_z * 2
        # only position variable
        self.dim_z = dim_z
        self.P = P
        self.R = R
        self.Q = Q
        self.dt = dt
        self.should_save = should_save

    def build(self, x=None):
        if x is None:
            x = np.zeros_like(self.dim_x)

        kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        kf.x = np.asarray(x)
        # state transition matrix
        kf.F = block_diagonal_array(self.dim_z, [[1., self.dt], [0., 1.]])
        # measurement function
        kf.H = block_diagonal_array(self.dim_z, [[1, 0]])

        # measurement uncertainty
        kf.R *= self.R
        # covariance matrix
        if np.isscalar(self.P):
            kf.P = self.P
        else:
            # [:] makes deep copy
            kf.P[:] = self.P
        if np.isscalar(self.Q):
            kf.Q *= Q_discrete_white_noise(dim=2, dt=self.dt,
                                           var=self.Q, block_size=self.dim_z)
        else:
            kf.Q[:] = self.Q

        self.filter = kf
        # Don't call save() here, save current state is the default behavior of Saver.
        self.saver = Saver(self.filter) if self.should_save else None

    def predict(self):
        self.filter.predict()

    def update(self, z):
        self.filter.predict()
        self.filter.update(z)

    def save(self):
        if self.should_save == False:
            print(
                "Warning: initial filter with `should_save=False`, but calling `save()`")
            return
        self.saver.save()

    @property
    def x(self):
        return self.filter.x.reshape((self.dim_z, 2))[:, 0]

    @property
    def dx(self):
        return self.filter.x.reshape((self.dim_z, 2))[:, 1]


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
    kf.x = np.asanyarray(x)
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
