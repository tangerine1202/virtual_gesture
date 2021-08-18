![21 hand landmarks](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

## Spec

### Tracking
- existence
  - shape: (1,)
  - range: [0, 1]
- handedness
  - shape: (1,)
  - range: [0, 1] (0: left, 1: right)
- landmarks
  - shape: (21, 3,)
  - range: [0, 1] (scale with `image_width`, `image_height`, `image_width`)

### Filter
- existence
  - *state shape: (2,) (position, velocity)*
  - *z shape: (1,) (position)*
  - x: [0.5, 0].T
  - F: [[1, dt], [0, 1]]
  - H: [1, 0]
  - init_P: eye(2,2) * 0.5
  - init_Q: eye(2,2) * 0 **(not sure)**
  - init_R: 0.2
- handedness
  - *state shape: (2,) (position, velocity)*
  - *z shape: (1,) (position)*
  - x: [0.5, 0].T
  - F: [[1, dt], [0, 1]]
  - H: [1, 0]
  - init_P: eye(2,2) * 0.5
  - init_Q: eye(2,2) * 0 **(not sure)**
  - init_R: 0.5 **(tuning)**
- landmarks
  - *state shape: (21 * 3 * 2,) = (126,) ((landmarks) (x,y,z) (position, velocity))*
  - *z shape: (21 * 3,) = (63,) ((landmarks) (x,y,z))*
  - x: [n=126, value=0.5].T
  - F: [[1, dt, 0, ...], [0, 1, dt, 0, ...], [0, 0, 1, dt, 0, ...] ... [dt, 0, ..., 1]]
  - H: [1, 0, 1, 0, ...]
  - init_P: eye(126,126) * 0.5
  - init_Q: eye(126,126) * 0 **(not sure)**
  - init_R: 0.01 **(should be low)**