"""Tests for rl.core.envs.synthetic_dataset."""

import unittest

import numpy as np

from rl.core.envs import synthetic_dataset


class SyntheticDatasetTest(unittest.TestCase):
  pass


class TileImageTest(unittest.TestCase):

  def setUp(self):
    self._tiles = [
      # C x H x W
      np.full((1, 2, 1), 1),
      np.full((1, 1, 3), 2),
    ]

  def test_tile_image(self):
    tile_xys = [(1, 2), (0, 1)]

    expected = np.array(
      [[
        [0, 0, 0],
        [2, 2, 2],
        [0, 1, 0],
        [0, 1, 0],
      ]]
    )
    actual = synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4)
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_overlapping(self):
    tile_xys = [(1, 1), (0, 1)]

    expected = np.array(
      [[
        [0, 0, 0],
        [2, 2, 2],  # at (1, 1), 2 overrides 1 (due to tiling order)
        [0, 1, 0],
        [0, 0, 0],
      ]]
    )
    actual = synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4)
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_out_of_bounds(self):
    tile_xys = [(1, 2), (1, 1)]

    with self.assertRaises(ValueError):
      synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4)
