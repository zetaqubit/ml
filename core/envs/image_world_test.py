"""Tests for rl.core.envs.image_world."""

import unittest

import numpy as np

from rl.core.envs import image_world


class ImageWorldEnvTest(unittest.TestCase):

  def setUp(self):
    # Shape [n, c, h, w] -> [2, 1, 3, 4]
    self._images = np.array([
      [[
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34],
      ]],
      [[
        [51, 52, 53, 54],
        [61, 62, 63, 64],
        [71, 72, 73, 74],
      ]],
    ])

  def test_step_in_bounds(self):
    env = image_world.ImageWorldEnv(self._images, window_size=2, seed=0)

    expected = self._images[0, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0.34)))
    np.testing.assert_array_equal(expected, env.step(action=(0.49, 0.66)))

    expected = self._images[0, :, 1:3, 1:3]
    np.testing.assert_array_equal(expected, env.step(action=(0.50, 0.67)))
    np.testing.assert_array_equal(expected, env.step(action=(0.74, 0.99)))

    expected = self._images[0, :, 1:3, 2:4]
    np.testing.assert_array_equal(expected, env.step(action=(0.75, 0.67)))
    np.testing.assert_array_equal(expected, env.step(action=(0.75, 0.99)))

  def test_step_padded(self):
    env = image_world.ImageWorldEnv(self._images, window_size=3, seed=0)

    expected = np.array([[
      [0, 0, 0],
      [0, 11, 12],
      [0, 21, 22],
    ]])
    np.testing.assert_array_equal(expected, env.step(action=(0, 0)))
    np.testing.assert_array_equal(expected, env.step(action=(0.24, 0.33)))

    expected = np.array([[
      [0, 0, 0],
      [11, 12, 13],
      [21, 22, 23],
    ]])
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0)))
    np.testing.assert_array_equal(expected, env.step(action=(0.49, 0.33)))

    expected = np.array([[
      [0, 0, 0],
      [13, 14, 0],
      [23, 24, 0],
    ]])
    np.testing.assert_array_equal(expected, env.step(action=(0.75, 0)))
    np.testing.assert_array_equal(expected, env.step(action=(0.99, 0.33)))

    expected = np.array([[
      [13, 14, 0],
      [23, 24, 0],
      [33, 34, 0],
    ]])
    np.testing.assert_array_equal(expected, env.step(action=(0.75, 0.34)))
    np.testing.assert_array_equal(expected, env.step(action=(0.99, 0.66)))

    expected = np.array([[
      [23, 24, 0],
      [33, 34, 0],
      [0, 0, 0],
    ]])
    np.testing.assert_array_equal(expected, env.step(action=(0.75, 0.67)))
    np.testing.assert_array_equal(expected, env.step(action=(0.99, 0.99)))

  def test_reset(self):
    env = image_world.ImageWorldEnv(self._images, window_size=2, seed=0)
    expected = self._images[0, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0.34)))

    env.reset()
    expected = self._images[1, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0.34)))

    env.reset()
    expected = self._images[1, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0.34)))

    env.reset()
    expected = self._images[0, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env.step(action=(0.25, 0.34)))


class TileImageTest(unittest.TestCase):

  def setUp(self):
    self._tiles = [
      # C x H x W
      np.full((1, 2, 1), 1),
      np.full((1, 1, 3), 2),
    ]

  def test_tile_image(self):
    tile_yxs = [(2, 1), (1, 0)]

    expected = np.array(
      [[
        [0, 0, 0],
        [2, 2, 2],
        [0, 1, 0],
        [0, 1, 0],
      ]]
    )
    actual = image_world.tile_image(self._tiles, tile_yxs, 4, 3)
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_overlapping(self):
    tile_yxs = [(1, 1), (1, 0)]

    expected = np.array(
      [[
        [0, 0, 0],
        [2, 2, 2],  # at (1, 1), 2 overrides 1 (due to tiling order)
        [0, 1, 0],
        [0, 0, 0],
      ]]
    )
    actual = image_world.tile_image(self._tiles, tile_yxs, 4, 3)
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_out_of_bounds(self):
    tile_yxs = [(2, 1), (1, 1)]

    with self.assertRaises(ValueError):
      image_world.tile_image(self._tiles, tile_yxs, 4, 3)
