"""A large image Gym environment.

A Gym environment where the world is a large image, observable to the agent
only through a small window. The agent chooses the spatial location (x, y) and
the zoom (z) at which to place the window. Zooming in provides a more detailed
but also more local view.

The challenge is to integrate detailed information from multiple locations and
make an image-level prediction. For example, imagine 10 random MNIST digits
scattered across the world, and the goal is to predict the sum.

This is intended to be a generalization of the Cluttered Translated MNIST
task [1].

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
"""

import gym
import gym.utils.seeding
import gym.spaces
import numpy as np
import torch as th


class ImageWorld(gym.core.Env):
  """An Image World environment.

  The environment holds a set of images. On each reset, an image is randomly
  chosen to the world the agent explores.

  The actions available to the agent are:
    a) observe(x, y, z): center the window at pixel (x, y) and zoom z. The
       agent observes the image patch through the window.

       The normalized coordinate system is defined as follows:
         (0, 0): window centered on top-left corner of the image
         (1, 1): window centered on bottom-right corner of the image
       Note that the corner coordinates, 3/4 of the patch will be 0-padded.

       TODO: define zoom z
       TODO: consider discrete vs continuous

    b) predict(c): predict the image-level characteristic of interest. The
       agent receives a reward based on the accuracy of its prediction. This
       concludes the episode, and the environment needs to be reset.
  """

  # Rewards
  REWARD_CORRECT = 0
  REWARD_INCORRECT = -10
  REWARD_GLIMPSE = -1

  def __init__(self, window_size, images, labels, num_classes):
    """Creates an ImageWorld.

    :param window_size: size of the square window, in pixels.
    :param images: set of images. numpy array shaped [n, c, h, w].
    """
    assert window_size > 0
    assert images.ndim == 4
    assert len(images) == len(labels)

    self._win_sz = window_size
    self._images = images
    self._labels = labels
    self._num_classes = num_classes
    self._n, self._c, self._h, self._w = images.shape
    self._images_mini = self._create_minimaps()

    self._rand = None
    self._current_image_index = None
    self.seed()

    # Gym-specific attributes
    self._action_space = gym.spaces.Dict({
      # Position window at (x, y, z).
      'window': gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float),

      # Whether to make a prediction.
      'should_predict': gym.spaces.Discrete(2),

      # Predicted category label
      'prediction': gym.spaces.Discrete(num_classes),
    })

    self._observation_space = gym.spaces.Box(
        low=0, high=1, shape=(self._c, self._win_sz, self._win_sz),
        dtype=np.float)

  @property
  def action_space(self):
    """Returns the action space.

    Action space:
      'window': [x, y, z] position to place the window. [0, 1]x[0, 1]x[0, 1]
      'should_predict': whether to make a prediction now and terminate the
         episode. {0, 1}
      'prediction': prediction of the class label. Only used if should_predict.
         {0, 1, ..., num_classes - 1}
    """
    return self._action_space

  @property
  def observation_space(self):
    """Returns the observation space.

    TODO: document
    TODO: always include downscaled 'minimap' view.
    """
    return self._observation_space

  @property
  def reward_range(self):
    """Returns the reward range.

    Rewards:
      - on each step: -1; episode continues.
      - is_predicting and correct: 0; episode terminates.
      - is_predicting and incorrect: -10; episode terminates.
    """
    return self.REWARD_INCORRECT, self.REWARD_CORRECT

  def step(self, action):
    """Moves the view window to (x, y) and returns the image patch there.

    :param action: the action to take, in format described by `action_space`.
    :return:
      observation (np.array): image patch visible through the window.
        Shape [c, win_sz, win_sz]. If done, this is None.
      reward (float): amount of reward returned after previous action
      done (boolean): whether the episode has ended, in which case `reset()`
        should be called.
      info (dict): contains auxiliary diagnostic information
    """
    window = action['window']
    should_predict = action['should_predict']
    prediction = action['prediction']

    if should_predict:
      assert 0 <= prediction < self._num_classes
      correct = prediction == self._labels[self._current_image_index]
      reward = self.REWARD_CORRECT if correct else self.REWARD_INCORRECT
      obs = None
      return obs, reward, True, {}

    assert len(window) == 2  # TODO: support zoom
    x, y = window
    obs = self.glimpse(x, y)
    return obs, self.REWARD_GLIMPSE, False, {}

  def glimpse(self, x, y):
    """Moves the view window to (x, y) and returns the image patch there.

    :param x: x-coordinates of the window, in range [0, 1).
    :param y: y-coordinates of the window, in range [0, 1).
    :return: image patch visible through the window. Shape [c, win_sz, win_sz].
    """
    image = self._images[self._current_image_index]
    images = np.expand_dims(image, axis=0)
    glimpses = glimpse_batch(th.from_numpy(np.array([x], dtype=np.float32)),
                             th.from_numpy(np.array([y], dtype=np.float32)),
                             th.from_numpy(images),
                             self._win_sz)
    return glimpses[0].numpy()

  def reset(self):
    """Resets the environment, loading a new image.

    :return:
      observation (np.array): minimap of the new image.
        Shape [c, win_sz, win_sz].
    """
    self._current_image_index = self._rand.randint(self._n)
    return self.minimap

  def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).

    :param seed: seed to use for all RNG in this environment. Default of
      None selects a random seed.
    :return: the random seed supplied (if any) or selected. Using this value
      to seed a future Env guarantees full reproducibility.
    """
    if seed is None:
      seed = gym.utils.seeding.create_seed(max_bytes=4)
    self._rand = np.random.RandomState(seed)
    self.reset()
    return [seed]

  @property
  def num_images(self):
    return len(self._images)

  @property
  def current_image_index(self):
    return self._current_image_index

  @property
  def image(self):
    return self._images[self._current_image_index]

  @property
  def label(self):
    return self._labels[self._current_image_index]

  @property
  def minimap(self):
    return self._images_mini[self._current_image_index]

  def _create_minimaps(self):
    minimap = np.empty((self._n, self._c, self._win_sz, self._win_sz))
    for i in range(self._n):
      minimap[i] = resize(self._images[i], self._win_sz, self._win_sz)
    return minimap


def resize(image, width, height):
  """Resizes image to specified size.

  :param image (np.array): input image, shaped [c, h, w].
  :param width: desired output width, in pixels
  :param height: desired output height, in pixels
  :return: (np.array) resized image, shaped [c, height, width].
  """
  import cv2

  orig_ndim = image.ndim

  # Convert CHW <-> WHC for OpenCV
  image = image.transpose(2, 1, 0)
  image = cv2.resize(image, (height, width))
  if orig_ndim != image.ndim:
    image = np.expand_dims(image, -1)  # Add channel back.
  image = image.transpose(2, 1, 0)
  return image


def glimpse_batch(x, y, images, win_sz, visualize=None):
  """Extracts glimpse patches centered at specified locations for each image.

  :param x: x-coordinates of the window, in range [0, 1). Shape [b].
  :param y: y-coordinates of the window, in range [0, 1). Shape [b].
  :param images: full-size images. Shaped [b, c, h, w]
  :param win_sz: size of the glimpse patch, in pixels.
  :return: image patch visible through the window. Shape [b, c, win_sz, win_sz].
  """
  assert ((0 <= x) & (x <= 1)).all()
  assert ((0 <= y) & (y <= 1)).all()

  batch, c, h, w = images.shape
  center_x, center_y = th.floor(x * w), th.floor(y * h)

  w_before = win_sz // 2
  w_after = win_sz - w_before
  l, r = center_x - w_before, center_x + w_after
  t, b = center_y - w_before, center_y + w_after

  zero = th.tensor(0).type_as(l)
  out_l, out_r = th.max(zero, -l), win_sz - th.max(zero, r - w)
  out_t, out_b = th.max(zero, -t), win_sz - th.max(zero, b - h)

  in_l, in_r, in_t, in_b = _clamp_to_bounds(l, r, t, b, w, h)

  def as_int(*tensors):
    return [x.int() for x in tensors]
  in_l, in_r, in_t, in_b = as_int(in_l, in_r, in_t, in_b)
  out_l, out_r, out_t, out_b = as_int(out_l, out_r, out_t, out_b)

  patch = th.zeros((batch, c, win_sz, win_sz)).type_as(images)
  for i in range(batch):
    patch[i, :, out_t[i]:out_b[i], out_l[i]:out_r[i]] = (
        images[i, :, in_t[i]:in_b[i], in_l[i]:in_r[i]])
    if visualize is not None:
      val_op = getattr(visualize, '__getitem__')
      val = 10 * val_op(i) if val_op else visualize
      images[i, :, in_t[i]:in_b[i], in_l[i]:in_r[i]] += val

  return patch


def _clamp_to_bounds(l, r, t, b, w, h):
  l = th.clamp(l, 0, w)
  r = th.clamp(r, 0, w)
  t = th.clamp(t, 0, h)
  b = th.clamp(b, 0, h)
  return l, r, t, b

