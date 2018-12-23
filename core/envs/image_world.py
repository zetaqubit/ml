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
import numpy as np


class ImageWorldEnv(gym.core.Env):
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

  def __init__(self, images, window_size, seed=0):
    """Creates an ImageWorldEnv.

    :param images: set of images. numpy array shaped [n, c, h, w].
    :param window_size: size of the square window, in pixels.
    :param seed: random seed controlling the sampling of images from the set.
    """
    assert images.ndim == 4
    assert window_size > 0

    self._images = images
    self._n, self._c, self._h, self._w = images.shape
    self._win_sz = window_size
    self._rand = np.random.RandomState(seed)
    self._current_image = None
    self.reset()

  def step(self, action):
    """Moves the view window to (x, y) and returns the image patch there.

    :param action: (x, y) coordinates of the window, in range [0, 1)x[0, 1)
    :return:
      observation:
        image patch visible through the window. Shape [c, win_sz, win_sz]
      TODO: define following
      reward (float): amount of reward returned after previous action
      done (boolean): whether the episode has ended, in which case further
        step() calls will return undefined results
      info (dict): contains auxiliary diagnostic information (helpful for
        debugging, and sometimes learning)

    """
    x, y = action
    assert 0 <= x <= 1 and 0 <= y <= 1

    image = self._images[self._current_image]

    center_x, center_y = int(x * self._w), int(y * self._h)

    w_before = self._win_sz // 2
    w_after = self._win_sz - w_before
    l, r = center_x - w_before, center_x + w_after
    t, b = center_y - w_before, center_y + w_after

    out_l, out_r = max(0, -l), self._win_sz - max(0, r - self._w)
    out_t, out_b = max(0, -t), self._win_sz - max(0, b - self._h)

    in_l, in_r, in_t, in_b = self._clamp_to_bounds(l, r, t, b)

    patch = np.zeros((self._c, self._win_sz, self._win_sz))
    patch[:, out_t:out_b, out_l:out_r] = image[:, in_t:in_b, in_l:in_r]
    return patch

  def _clamp_to_bounds(self, l, r, t, b):
    l = np.clip(l, 0, self._w)
    r = np.clip(r, 0, self._w)
    t = np.clip(t, 0, self._h)
    b = np.clip(b, 0, self._h)
    return l, r, t, b

  def reset(self):
    self._current_image = self._rand.randint(self._n)


def tile_image(tiles, tile_yxs, height, width):
  """Creates an image by blitting tiles at specified locations.

  :param tiles: list of np.arrays shaped [c, h, w]
  :param tile_yxs: corresponding list of locations to place the tiles. These
    are the output coordinates of the top-left corners of the tiles.
    Format: [(y_1, x_1), (y_2, x_2), ...]
  :param height: height of the output image, in pixels
  :param width: width of the output image, in pixels
  :return: the merged image, as an np.array shaped [c, height, width]
  """
  channels = tiles[0].shape[0]
  image = np.zeros((channels, height, width))
  for tile, (y, x) in zip(tiles, tile_yxs):
    tile_c, tile_h, tile_w = tile.shape
    image[:, y:y+tile_h, x:x+tile_w] = tile
  return image