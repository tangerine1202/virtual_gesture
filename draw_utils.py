from utils import dotdict
import Hand

import cv2
import numpy as np

COLOR = dotdict({
    'black': (0, 0, 0),
    'gray':	(128, 128, 128),
    'silver':	(192, 192, 192),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'yellow':	(255, 255, 0),
    'lime':	(0, 255, 0),
    'cyan':	(0, 255, 255),
    'blue':	(0, 0, 255),
    'magenta': (255, 0, 255),
})


def draw_handedness(img, handedness):
    if handedness[0] < .5:
        label = 'Left'
    elif handedness[0] > .5:
        label = 'Right'
    else:
        label = 'Unknown'

    img_w, img_h = img.shape[1], img.shape[0]
    cv2.putText(img, f'{label} {np.abs((handedness[0]-.5)*2) :.2f}',
                org=(img_w//2 - 200, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=COLOR.red,
                lineType=2)


def draw_landmarks(img, landmarks, color=(50, 255, 50)):
    # draw connection
    pairs = []
    # wrist to index~pinky
    pairs.append((0, 1, COLOR.silver))
    pairs.append((0, 5, COLOR.silver))
    pairs.append((0, 17, COLOR.silver))
    # finger mcp to finger mcp
    pairs.append((5, 9, COLOR.silver))
    pairs.append((9, 13, COLOR.silver))
    pairs.append((13, 17, COLOR.silver))
    # mcp to tip
    for i, c in zip(range(1, 21, 4), [COLOR.magenta, COLOR.lime, COLOR.cyan, COLOR.blue, COLOR.yellow]):
        for j in range(3):
            pairs.append((i+j, i+j+1, c))

    for pp1, pp2, c in pairs:
        p1 = landmarks[pp1, 0:2, 0]
        p2 = landmarks[pp2, 0:2, 0]
        cv2.line(img, p1.astype(int), p2.astype(int), thickness=2, color=c)

    # draw dots
    for i in range(21):
        p = landmarks[i, 0:2, 0]
        cv2.circle(img, p.astype(int), radius=3,
                   thickness=-1, color=COLOR.red)


def draw_finger_state(img, handedness, finger_states):
    img_w, img_h = img.shape[1], img.shape[0]

    state_str = ''
    for finger_idx in range(5):
        if handedness[0] < 0.5:
            state_str = f'{finger_states[finger_idx].value}{state_str}'
        else:
            state_str = f'{state_str}{finger_states[finger_idx].value}'
    cv2.putText(img, state_str,
                org=(img_w//2, 30),  # bottomLeftCornerOfText
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=COLOR.red,
                lineType=2)


def draw_click_drag(img, landmarks, is_click, is_drag):
    # index tip
    p = landmarks[8, 0:2, 0]

    # draw click
    if is_click:
        cv2.circle(img, p.astype(int),
                   radius=7, thickness=1, color=COLOR.white)

    # draw drag
    if is_drag:
        cv2.circle(img, p.astype(int),
                   radius=7, thickness=1, color=COLOR.white)
        cv2.circle(img, p.astype(int),
                   radius=10, thickness=1, color=COLOR.white)


def draw_3axis(img, rotation_matrix, delta_origin):
    """
    Parameters
    ---
    img : np.array
      The image annotated to.
    rotation_matrix : np.array
      Rotation matrix used to transform.
    delta_origin : np.array
      Offset to original wrist.
    """
    start = Hand.Hand.reverse_landmarks_transformation(
        [0, 0, 0], rotation_matrix, delta_origin)
    x_end = Hand.Hand.reverse_landmarks_transformation(
        [40, 0, 0], rotation_matrix, delta_origin)
    y_end = Hand.Hand.reverse_landmarks_transformation(
        [0, 40, 0], rotation_matrix, delta_origin)
    z_end = Hand.Hand.reverse_landmarks_transformation(
        [0, 0, 40], rotation_matrix, delta_origin)
    cv2.arrowedLine(img, (start[0:2]).astype(
        int), (x_end[0:2]).astype(int), thickness=2, color=COLOR.black)
    cv2.arrowedLine(img, (start[0:2]).astype(
        int), (y_end[0:2]).astype(int), thickness=2, color=COLOR.gray)
    cv2.arrowedLine(img, (start[0:2]).astype(
        int), (z_end[0:2]).astype(int), thickness=2, color=COLOR.blue)
    cv2.putText(img, 'x', org=((x_end[0:2] + [5, 5]).astype(int)), color=COLOR.black,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.7, lineType=2)
    cv2.putText(img, 'y', org=((y_end[0:2] + [5, 5]).astype(int)), color=COLOR.gray,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.7, lineType=2)
    cv2.putText(img, 'z', org=((z_end[0:2] + [5, 5]).astype(int)), color=COLOR.blue,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.7, lineType=2)
