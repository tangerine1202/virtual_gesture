from config import REAL_BONES_LENGTH
from utils import *

import platform
import numpy as np
from filterpy.common import Saver

np.set_printoptions(precision=4)


class Hand:
    def __init__(self, image_width, image_hight, should_saves=[], MATCH_REAL_BONES_LEGNTH=False):
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
        vel_P = (self._image_width/4, self._image_hight /
                 4, self._image_depth/4)

        # landmarks Q
        if MATCH_REAL_BONES_LEGNTH:
            pos_Q = (35./3, 40./3, 25./3)
            vel_Q = (750/3, 750/3, 500/3)
            Q_corr = self._get_landmarks_Q_corr(MATCH_REAL_BONES_LEGNTH)
        else:
            pos_Q = (.25, .25, 1.25)
            vel_Q = (52.5, 54., 12.5)
            Q_corr = self._get_landmarks_Q_corr()

        # landmarks R
        if MATCH_REAL_BONES_LEGNTH:
            pos_R = (5., 5., 10.)
        else:
            # TODO: need update to image depth scale
            pos_R = (1.5, 1.5, 5.)

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

    def build(self, z_existence=None, z_handedness=None, z_landmarks=None):
        """
        Build the filters (and saver) for hand tracking.

        Parameters
        ---
        z_existence : scalar, None
            Optional. Initial measurement of existence using to initialize observable state.
        z_handedness : scalar, None
            Optional. Initial measurement of handedness using to initialize observable state.
        z_landmarks : np.array, None
            Optional. Initial measurement of landmarks using to initialize observable state.
        """
        if z_existence is None:
            existence_x = self._et_x
        else:
            assert np.isscalar(z_existence)
            existence_x = [z_existence, self._et_x[1]]
        if z_handedness is None:
            handedness_x = self._hn_x
        else:
            assert np.isscalar(z_handedness)
            handedness_x = [z_handedness, self._hn_x[1]]
        if z_landmarks is None:
            landmarks_x = self._lm_x
        else:
            assert z_landmarks.shape == (21, 3)
            landmarks_x = np.stack(
                [z_landmarks, self._lm_x.reshape(21, 3, 2)[:, :, 1]], axis=2).flatten()

        # Build existence filter
        self._existence_f = pos_vel_filter(
            x=existence_x, P=self._et_P, R=self._et_R, Q=self._et_Q, dt=self._dt)
        # Build handedness filter
        self._handedness_f = pos_vel_filter(
            x=handedness_x, P=self._hn_P, R=self._hn_R, Q=self._hn_Q, dt=self._dt)
        # Build landmarks filters
        self._landmarks_f = multi_pos_vel_filter(
            x=landmarks_x, P=self._lm_P, R=self._lm_R, Q=self._lm_Q, dt=self._dt)

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

    def update(self, z_existence=None, z_handedness=None, z_landmarks=None):
        if np.isscalar(z_existence):
            self._existence_f.update(z_existence)

        if np.isscalar(z_handedness):
            self._handedness_f.update(z_handedness)

        if type(z_landmarks) == np.ndarray and z_landmarks.size != 0:
            # switch hand landmarks if the sensor detects wrong handedness
            should_switch_hand = (self._handedness_f.x[0] > .5 and self._handedness_f.z < .5) or (
                self._handedness_f.x[0] < .5 and self._handedness_f.z > .5)
            if should_switch_hand:
                z_landmarks = self._switch_hand(z_landmarks)

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

    # DEV:
    @staticmethod
    def match_real_bones_length(landmarks):
        assert landmarks.shape == (21, 3)
        res = np.empty_like(landmarks)
        res[0] = landmarks[0]
        for finger_name in REAL_BONES_LENGTH.keys():
            index = REAL_BONES_LENGTH[finger_name]['index']
            length = REAL_BONES_LENGTH[finger_name]['length']
            for i in range(0, 4):
                idx_from = index[i]
                idx_to = index[i+1]
                unit = (landmarks[idx_to] - landmarks[idx_from]) / \
                    np.linalg.norm(landmarks[idx_to] - landmarks[idx_from])
                res[idx_to] = res[idx_from] + unit * length[i]

        return res

    @staticmethod
    def transform_to_palm_coordinate(landmarks, is_right_hand):
        """
        Using wrist as origin, vector from index-mcp to wrist as y-axis, cross vector of pinky-mcp to thumb and y-axis as z-axis

        Parameters
        ---
        landmarks : np.array
          Original landmarks.
        is_right_hand : bool
          Used to determine axis

        Returns
        ---
        transformed_landmarks : np.array
          transformed landmarks
        rotation_matrix : np.array
          Rotation matrix used to transform to new coordinate.
          Needed to reverse transformation.
        delta_origin : np.array
          Offset of original wrist.
          Needed to reverse transformation.
        """
        assert landmarks.shape == (21, 3)
        landmarks = np.array(landmarks)

        # shift origin
        delta_origin = np.copy(landmarks[0])
        landmarks -= delta_origin

        # calculate rotation matrix
        axis_y = landmarks[0] - landmarks[5]
        if is_right_hand:
            axis_z = np.cross(landmarks[0] - landmarks[9], axis_y)
        else:
            axis_z = np.cross(axis_y, landmarks[0] - landmarks[9])
        r = get_coordinate_rotation_matrix(
            [axis_y, axis_z], [[0, 1, 0], [0, 0, 1]])

        transformed_landmarks = landmarks @ r

        return transformed_landmarks, r, delta_origin

    @staticmethod
    def reverse_landmarks_transformation(transformed_landmarks, rotation_matrix, delta_origin):
        """
        Transform landmarks back to original coordinate.

        Parameters
        ---
        transformed_landmarks : np.array
          landmarks needed to be transformed back.
        rotation_matrix : np.array
          Rotation matrix used to transform before.
        delta_origin : np.array
          Offset to original wrist.

        Returns
        ---
        landmarks : np.array
          reversed landmarks
        """
        transformed_landmarks = np.array(transformed_landmarks)

        # NOTE: invsere of rotation matrix = transpose of rotation matrix
        landmarks = transformed_landmarks @ rotation_matrix.T + delta_origin

        return landmarks

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

    def _switch_hand(self, landmarks):
        landmarks = np.array(landmarks)
        landmarks[[1, 2, 3, 4, 5, 6, 7, 8]] = landmarks[[
            17, 18, 19, 20, 13, 14, 15, 16]]
        return landmarks

    def _get_landmarks_Q_corr(self, MATCH_REAL_BONES_LENGTH=False):
        if MATCH_REAL_BONES_LENGTH:
            return np.array([
                [0.3674, 0.2007, 0.0001],
                [0.3467, 0.2538, 0.2373],
                [0.4352, 0.248,  0.4594],
                [0.6474, 0.2478, 0.6132],
                [1.    , 0.3055, 0.7707],
                [0.496,  0.2061, 0.4993],
                [0.5867, 0.2266, 0.7067],
                [0.6797, 0.4425, 0.8383],
                [0.8009, 0.9155, 1.],
                [0.499,  0.1951, 0.4528],
                [0.5932, 0.248,  0.7334],
                [0.6425, 0.5645, 0.838],
                [0.6973, 1.    , 0.9354],
                [0.508,  0.2001, 0.4582],
                [0.6014, 0.2713, 0.7429],
                [0.6377, 0.5462, 0.7993],
                [0.6902, 0.9347, 0.8386],
                [0.5273, 0.2152, 0.4503],
                [0.5935, 0.2589, 0.6079],
                [0.615,  0.3547, 0.6598],
                [0.6642, 0.6032, 0.7364],
            ])
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
    print('build without initial measurement successfully')
    hand.reset()
    print('reset successfully')
    hand.build(0, .5, np.arange(21*3).reshape(21, 3))
    print('build with initial measurement successfully')
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
