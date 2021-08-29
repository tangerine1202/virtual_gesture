from utils import *

import platform
import numpy as np
from filterpy.common import Saver

np.set_printoptions(precision=4)


class Hand:
    def __init__(self, image_width, image_hight, should_saves=[]):
        """
        Parameters
        ----------
        image_width : scalar
          image width.

        image_hight : scalar
          image hight.

        should_saves: array or None
          Optionally, array of name of saved filters, allowed contain ['existence', 'handedness', 'landmarks']
        """

        # FUTURE: may have a better scale factor
        self._image_width = image_width
        self._image_hight = image_hight
        self._image_depth = image_width

        self._should_saves = should_saves

        self._dt = 1./15

        self._et_x = (.5, 0.)
        self._et_P = 1.
        self._et_R = .25
        self._et_Q = .5

        self._hn_x = (.5, 0.)
        self._hn_P = 1.
        self._hn_R = .3
        self._hn_Q = .5

        # landmarks x
        pos_x = (self._image_width/2, self._image_hight/2, 0.)
        vel_x = (0., 0., 0.)

        # landmarks P
        pos_P = (self._image_width/2, self._image_hight /
                 2, self._image_depth/2)
        vel_P = (self._image_width/2, self._image_hight /
                 2, self._image_depth/2)

        # landmarks Q
        pos_Q = (.25, .25, 1.25)
        vel_Q = (52.5, 54., 12.5)
        Q_corr = self.__get_landmarks_Q_corr()

        # landmarks R
        pos_R = (.25, .25, .75)

        self._lm_x = np.tile(
            np.stack([pos_x, vel_x], axis=1), (21, 1, 1)).flatten()
        self._lm_P = np.eye(
            126) * np.tile(np.stack([pos_P, vel_P], axis=1), (21, 1, 1)).flatten()
        lm_pos_Q = np.tile(pos_Q, [21, 1])
        lm_vel_Q = vel_Q * Q_corr
        self._lm_Q = np.eye(
            126) * np.stack([lm_pos_Q, lm_vel_Q], axis=2).flatten()
        lm_R_block = np.eye(3) * pos_R
        self._lm_R = block_diagonal_array(63//3, lm_R_block)

    def build(self):
        # Build existence filter
        self._existence_f = pos_vel_filter(
            x=self._et_x, P=self._et_P, R=self._et_R, Q=self._et_Q, dt=self._dt)
        # Build handedness filter
        self._handedness_f = pos_vel_filter(
            x=self._hn_x, P=self._hn_P, R=self._hn_R, Q=self._hn_Q, dt=self._dt)
        # Build landmarks filters
        self._landmarks_f = multi_pos_vel_filter(
            self._lm_x, P=self._lm_P, R=self._lm_R, Q=self._lm_Q, dt=self._dt)

        if 'existence' in self._should_saves:
            self._existence_s = Saver(self._existence_f)
        if 'handedness' in self._should_saves:
            self._handedness_s = Saver(self._handedness_f)
        if 'landmarks' in self._should_saves:
            self._landmarks_s = Saver(self._landmarks_f)

    def predict(self):
        self._existence_f.predict()
        self._handedness_f.predict()
        self._landmarks_f.predict()

    def update(self, z_existence, z_handedness=None, z_landmarks=None):
        self._existence_f.update(z_existence)

        if type(z_handedness) == np.ndarray and z_handedness.size != 0:
            self._handedness_f.update(z_handedness)

        if type(z_landmarks) == np.ndarray and z_landmarks.size != 0:
            # switch hand landmarks if the sensor detects wrong handedness
            should_switch_hand = (self._handedness_f.x[0] > .5 and self._handedness_f.z < .5) or (
                self._handedness_f.x[0] < .5 and self._handedness_f.z > .5)
            if should_switch_hand:
                z_landmarks = self.__switch_hand(z_landmarks)

            # scale to image size
            z_landmarks *= [self._image_width,
                            self._image_hight, self._image_depth]

            # FUTURE: change coordinate
            # z_shift_landmarks, rotation_matrix, delta_origin = shift_to_palm_coordinate(z_landmarks, handedness[0] > 0.5)
            # landmarks_f.update(z_shift_landmarks.flatten())

            # flatten
            z_landmarks = z_landmarks.flatten()

            # update landmarks
            self._landmarks_f.update(z_landmarks)

    def save(self):
        if 'existence' in self._should_saves:
            self._existence_s.save()
        if 'handedness' in self._should_saves:
            self._handedness_s.save()
        if 'landmarks' in self._should_saves:
            self._landmarks_s.save()

    def reset(self):
        if 'existence' in self._should_saves:
            del self._existence_s
        if 'handedness' in self._should_saves:
            del self._handedness_s
        if 'landmarks' in self._should_saves:
            del self._landmarks_s

        del self._existence_f
        del self._handedness_f
        del self._landmarks_f

        self.build()

    @property
    def existence_x(self):
        return self._existence_f.x

    @property
    def handedness_x(self):
        return self._handedness_f.x

    @property
    def landmarks_x(self):
        return self._landmarks_f.x.reshape(21, 3, 2)

    @property
    def existence_s(self):
        if hasattr(self, '_existence_s'):
            return self._existence_s
        return None

    @property
    def handedness_s(self):
        if hasattr(self, '_handedness_s'):
            return self._handedness_s
        return None

    @property
    def landmarks_s(self):
        if hasattr(self, '_landmarks_s'):
            return self._landmarks_s
        return None

    @property
    def dt(self):
        return self._dt

    def __switch_hand(self, landmarks):
        landmarks = np.array(landmarks)
        landmarks[[1, 2, 3, 4, 5, 6, 7, 8]] = landmarks[[
            17, 18, 19, 20, 13, 14, 15, 16]]
        return landmarks

    def __get_landmarks_Q_corr(self):
        if platform.system() == 'Windows':
            return np.array([
                [0.3511, 0.349, 0.0004],
                [0.5964, 0.3559, 0.2361],
                [0.7705, 0.3104, 0.4081],
                [0.8716, 0.3043, 0.5259],
                [1., 0.3614, 0.6697],
                [0.5417, 0.2475, 0.5229],
                [0.6963, 0.2957, 0.7271],
                [0.8096, 0.5599, 0.8767],
                [0.9454, 0.964, 1.],
                [0.4744, 0.2455, 0.4395],
                [0.6485, 0.3047, 0.7169],
                [0.6545, 0.591, 0.8171],
                [0.6612, 1., 0.8971],
                [0.4781, 0.2569, 0.358],
                [0.6344, 0.3066, 0.5761],
                [0.6231, 0.465, 0.6243],
                [0.6208, 0.7028, 0.6646],
                [0.487, 0.2692, 0.3178],
                [0.6103, 0.2947, 0.456],
                [0.6283, 0.3491, 0.5105],
                [0.6442, 0.4335, 0.5658],
            ])
        else:
            return np.array([
                [0.3029, 0.288, 0.0001],
                [0.3697, 0.3071, 0.2132],
                [0.4982, 0.341, 0.3471],
                [0.6223, 0.3485, 0.4628],
                [0.7401, 0.3523, 0.6022],
                [0.5629, 0.325, 0.444],
                [0.725, 0.3822, 0.6111],
                [0.8593, 0.5839, 0.7724],
                [1., 0.9228, 0.9337],
                [0.5521, 0.2954, 0.4395],
                [0.697, 0.3602, 0.6971],
                [0.7584, 0.6306, 0.8509],
                [0.8241, 1., 1.],
                [0.5265, 0.2729, 0.4513],
                [0.6577, 0.3198, 0.6869],
                [0.7236, 0.4839, 0.7723],
                [0.7885, 0.6997, 0.8584],
                [0.4793, 0.2586, 0.4784],
                [0.5691, 0.2762, 0.6415],
                [0.6267, 0.3144, 0.7468],
                [0.6858, 0.3841, 0.8618]
            ])


if __name__ == '__main__':
    hand = Hand(720, 720, should_saves=['landmarks'])
    print('init successfully')
    hand.build()
    print('build successfully')
    hand.predict()
    print('predict successfully')
    hand.update(1, 1, np.arange(21*3).reshape(21, 3))
    print('update successfully')
    hand.save()
    print('save successfully')

    et_x = hand.existence_x
    hn_x = hand.handedness_x
    lm_x = hand.landmarks_x
    assert type(et_x) == np.ndarray
    assert type(hn_x) == np.ndarray
    assert type(lm_x) == np.ndarray
    print(lm_x.shape)
    print('get x successfully')

    et_s = hand.existence_s
    hn_s = hand.handedness_s
    lm_s = hand.landmarks_s
    assert et_s == None
    assert hn_s == None
    assert type(lm_s) == Saver
    print(np.array(lm_s.x_post).shape)
    print('get s successfully')
