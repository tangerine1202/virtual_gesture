from filter.FilterABC import PosVelFilter
from utils import *
import time
import math


class OneEuroFilter(PosVelFilter):
    def __init__(self, min_cutoff=1., beta=0., d_cutoff=1., should_save=False):
        """
        Decreasing the minimum cutoff frequency decreases slow speed jitter. Increasing the speed coefficient decreases speed lag.

        Parameters
        --
        min_cutoff: scalar
            The minimum cutoff frequency.
        beta: scalar
            The speed coefficient.
        """
        self.min_cutoff = np.float32(min_cutoff)
        self.beta = np.float32(beta)
        self.d_cutoff = np.float32(d_cutoff)

        self.should_save = should_save

    def build(self, x, dx=0., t=None):
        if t is None:
            t = time.time()
        self._t_prev = t

        self._x_prev = np.array(x)

        if np.isscalar(dx):
            self._dx_prev = np.full_like(self._x_prev, dx)
        else:
            self._dx_prev = dx

        self.saver = None
        if self.should_save:
            self.saver = dotdict({})
            self.saver.z = np.array(self._x_prev)[np.newaxis, ...]
            self.saver.x = np.array(self._x_prev)[np.newaxis, ...]
            self.saver.dx = np.array(self._dx_prev)[np.newaxis, ...]
            self.saver.t = np.array([self._t_prev])

    def predict(self, t=None):
        if t is None:
            t = time.time()
        dt = t - self._t_prev
        x_hat = self._x_prev + dt*self._dx_prev
        dx_hat = self._dx_prev

        # Memorize the previous values.
        self._s_z = None
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        pass

    def update(self, z, t=None):
        if t is None:
            t = time.time()
        z = np.asanyarray(z)

        dt = t - self._t_prev

        # The filtered derivative of the signal.
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = (z - self._x_prev) / dt
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(dt, cutoff)
        x_hat = self._exponential_smoothing(a, z, self._x_prev)

        # Memorize the previous values.
        self._s_z = z
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def save(self):
        self.saver.z = np.concatenate(
            [self.saver.z, self._s_z[np.newaxis, ...]], axis=0)
        self.saver.x = np.concatenate(
            [self.saver.x, self._x_prev[np.newaxis, ...]], axis=0)
        self.saver.dx = np.concatenate(
            [self.saver.dx, self._dx_prev[np.newaxis, ...]], axis=0)
        self.saver.t = np.append(self.saver.t, self._t_prev)

    @property
    def x(self):
        return self._x_prev

    @property
    def dx(self):
        return self._dx_prev

    def _smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev


if __name__ == '__main__':
    z = np.arange(21*3).reshape(21, 3)
    oneeuro = OneEuroFilter(1., 1., 1., should_save=True)
    print('init successfully')
    oneeuro.build(z, 0.)
    print('build successfully')
    oneeuro.predict()
    print('predict successfully')
    z = np.arange(1, 1+21*3).reshape(21, 3)
    oneeuro.update(z)
    print('update successfully')
    oneeuro.save()
    print('save successfully')

    x = oneeuro.x
    dx = oneeuro.dx
    print('get x, dx successfully')

    saver = oneeuro.saver
    sx = saver.x
    sdx = saver.dx
    assert np.all(x == sx[-1])
    assert np.all(dx == sdx[-1])
    print('get s successfully')
