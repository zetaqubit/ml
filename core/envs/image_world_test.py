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
    ], dtype=np.float)

    self._labels = np.array([1, 3])
    self._num_classes = 3

  def _create_env(self, window_size):
    return image_world.ImageWorldEnv(
        window_size=window_size, images=self._images, labels=self._labels,
        num_classes=self._num_classes)

  def _create_action(self, should_predict=None, window=None, pred=None):
    return {
      'should_predict': should_predict,
      'window': window,
      'prediction': pred
    }

  def assert_step_output_equal(self, expected, actual):
    self.assertEqual(len(expected), len(actual))
    np.testing.assert_array_equal(expected[0], actual[0])
    self.assertEqual(expected[1:], actual[1:])

  def test_glimpse_in_bounds(self):
    env = self._create_env(window_size=2)
    env.seed(0)

    expected = self._images[0, :, 0:2, 0:2]
    np.testing.assert_array_equal(expected, env._glimpse(0.25, 0.34))
    np.testing.assert_array_equal(expected, env._glimpse(0.49, 0.66))

    expected = self._images[0, :, 1:3, 1:3]
    np.testing.assert_array_equal(expected, env._glimpse(0.50, 0.67))
    np.testing.assert_array_equal(expected, env._glimpse(0.74, 0.99))

    expected = self._images[0, :, 1:3, 2:4]
    np.testing.assert_array_equal(expected, env._glimpse(0.75, 0.67))
    np.testing.assert_array_equal(expected, env._glimpse(0.75, 0.99))

  def test_glimpse_padded(self):
    env = self._create_env(window_size=3)
    env.seed(0)

    expected = np.array([[
      [0, 0, 0],
      [0, 11, 12],
      [0, 21, 22],
    ]])
    np.testing.assert_array_equal(expected, env._glimpse(0, 0))
    np.testing.assert_array_equal(expected, env._glimpse(0.24, 0.33))

    expected = np.array([[
      [0, 0, 0],
      [11, 12, 13],
      [21, 22, 23],
    ]])
    np.testing.assert_array_equal(expected, env._glimpse(0.25, 0))
    np.testing.assert_array_equal(expected, env._glimpse(0.49, 0.33))

    expected = np.array([[
      [0, 0, 0],
      [13, 14, 0],
      [23, 24, 0],
    ]])
    np.testing.assert_array_equal(expected, env._glimpse(0.75, 0))
    np.testing.assert_array_equal(expected, env._glimpse(0.99, 0.33))

    expected = np.array([[
      [13, 14, 0],
      [23, 24, 0],
      [33, 34, 0],
    ]])
    np.testing.assert_array_equal(expected, env._glimpse(0.75, 0.34))
    np.testing.assert_array_equal(expected, env._glimpse(0.99, 0.66))

    expected = np.array([[
      [23, 24, 0],
      [33, 34, 0],
      [0, 0, 0],
    ]])
    np.testing.assert_array_equal(expected, env._glimpse(0.75, 0.67))
    np.testing.assert_array_equal(expected, env._glimpse(0.99, 0.99))

  def test_step_glimpse(self):
    env = self._create_env(window_size=2)
    env.seed(0)

    action = self._create_action(should_predict=0, window=(0.25, 0.34))
    expected = (self._images[0, :, 0:2, 0:2], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action))

    action = self._create_action(should_predict=0, window=(0.50, 0.67))
    expected = (self._images[0, :, 1:3, 1:3], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action))

  def test_step_predict_correct(self):
    env = self._create_env(window_size=2)
    env.seed(0)

    action = self._create_action(should_predict=1, pred=1)
    expected = (None, 0, True, {})
    self.assert_step_output_equal(expected, env.step(action))

  def test_step_predict_incorrect(self):
    env = self._create_env(window_size=2)
    env.seed(0)

    action = self._create_action(should_predict=1, pred=2)
    expected = (None, -10, True, {})
    self.assert_step_output_equal(expected, env.step(action))

  def test_reset(self):
    env = self._create_env(window_size=2)
    env.seed(0)

    action_glimpse = self._create_action(should_predict=0, window=(0.25, 0.34))
    action_predict = self._create_action(should_predict=1, pred=1)

    expected = (self._images[0, :, 0:2, 0:2], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action_glimpse))
    expected = (None, 0, True, {})
    self.assert_step_output_equal(expected, env.step(action_predict))

    env.reset()
    expected = (self._images[1, :, 0:2, 0:2], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action_glimpse))
    expected = (None, -10, True, {})
    self.assert_step_output_equal(expected, env.step(action_predict))

    env.reset()
    expected = (self._images[1, :, 0:2, 0:2], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action_glimpse))
    expected = (None, -10, True, {})
    self.assert_step_output_equal(expected, env.step(action_predict))

    env.reset()
    expected = (self._images[0, :, 0:2, 0:2], -1, False, {})
    self.assert_step_output_equal(expected, env.step(action_glimpse))
    expected = (None, 0, True, {})
    self.assert_step_output_equal(expected, env.step(action_predict))

  def test_seed_specified_new_env(self):
    num_trials = 10
    num_resets = 20
    image_indices = np.empty((num_trials, num_resets))
    for t in range(num_trials):
      env = self._create_env(window_size=2)
      env.seed(1337)
      for n in range(num_resets):
        image_indices[t, n] = env.current_image_index
        env.reset()
    for t in range(1, num_trials):
      np.testing.assert_array_equal(image_indices[0], image_indices[t])

  def test_seed_returned_allows_reproducibility(self):
    env = self._create_env(window_size=2)
    num_trials = 10
    num_resets = 20
    image_indices = np.empty((num_trials, num_resets))
    for t in range(num_trials):
      if t == 0:
        chosen_seed = env.seed()[0]
      else:
        env.seed(chosen_seed)
      for n in range(num_resets):
        image_indices[t, n] = env.current_image_index
        env.reset()

    for t in range(num_trials):
      np.testing.assert_array_equal(image_indices[0], image_indices[t])

  def test_random_seed(self):
    num_trials = 10
    num_resets = 20
    image_indices = np.empty((num_trials, num_resets))
    for t in range(num_trials):
      env = self._create_env(window_size=2)
      for n in range(num_resets):
        image_indices[t, n] = env.current_image_index
        env.reset()

    unique_rows, counts = np.unique(image_indices, axis=0, return_counts=True)
    np.testing.assert_array_equal(
        np.ones(num_trials), counts,
        f'trials: \n{image_indices}\nunique:\n{unique_rows}')

  def test_minimap(self):
    images = np.array([
      [[
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
      ]],
      [[
        [0, 1, 0, 1],
        [0, 0, 3, 0],
        [1, 2, 9, 5],
        [3, 4, 5, 1],
      ]],
    ], dtype=np.float)
    env = image_world.ImageWorldEnv(
        window_size=2, images=images, labels=self._labels,
        num_classes=self._num_classes)
    env.seed(0)

    expected = np.array([
      [[
        [0, 1],
        [2, 3],
      ]],
      [[
        [0.25, 1],
        [2.5, 5],
      ]]
    ], dtype=np.float)
    np.testing.assert_array_equal(expected[0], env.minimap)

    env.reset()
    np.testing.assert_array_equal(expected[1], env.minimap)


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
