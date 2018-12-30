"""Tests for rl.core.envs.synthetic_dataset."""

import itertools
import unittest

import numpy as np

from rl.core.envs import synthetic_dataset


class SyntheticDatasetTest(unittest.TestCase):

  def setUp(self):
    self._tile_images = np.stack([
      # C x H x W
      np.full((1, 3, 2), 1),
      np.full((1, 3, 2), 2),
      np.full((1, 3, 2), 3),
    ])

    self._tile_labels = np.array([1, 10, 100])

  def test_one_tile_small_image(self):
    tiles = np.random.rand(1, 2, 4, 3)  # [n, c, h, w]
    labels = np.array([7])

    ds = synthetic_dataset.SyntheticDataset(3, 4, 10, 1, tiles, labels)

    np.testing.assert_array_equal(np.repeat(tiles, 10, axis=0), ds.images)
    np.testing.assert_array_equal(np.repeat(labels, 10), ds.labels)

  def test_one_tile_large_image(self):
    tiles = np.full((1, 1, 2, 3), 7)  # [n, c, h, w]
    labels = np.array([7])

    # Generate enough images that all possible tile positions occur in the 5x3.
    ds = synthetic_dataset.SyntheticDataset(5, 3, 100, 1, tiles, labels)

    image = np.array([
      [
        [7, 7, 7, 0, 0],
        [7, 7, 7, 0, 0],
        [0, 0, 0, 0, 0],
      ],
    ])
    possible_images = self._combinations_xy_shift(image, [0, 1, 2], [0, 1])
    actual_images = np.unique(ds.images, axis=0)
    np.testing.assert_array_equal(np.sort(possible_images, axis=0),
                                  np.sort(actual_images, axis=0))
    np.testing.assert_array_equal(np.repeat(labels, 100), ds.labels)

  @staticmethod
  def _combinations_xy_shift(self, image, x_shifts, y_shifts):
    images = []
    for x_shift in x_shifts:
      x_shifted = np.roll(image, x_shift, axis=-1)
      for y_shift in y_shifts:
        images.append(np.roll(x_shifted, y_shift, axis=-2))
    return np.stack(images)

  def test_label_fn(self):
    # Generate enough images that all possible combinations of the labels
    # are present.
    ds = synthetic_dataset.SyntheticDataset(
        4, 4, 100, 2, self._tile_images, self._tile_labels, label_fn=np.sum)
    expected_labels = {2, 11, 20, 101, 110, 200}
    self.assertEqual(expected_labels, set(ds.labels.flat))

    ds = synthetic_dataset.SyntheticDataset(
        4, 4, 100, 2, self._tile_images, self._tile_labels, label_fn=np.min)
    expected_labels = {1, 10, 100}
    self.assertEqual(expected_labels, set(ds.labels.flat))

    ds = synthetic_dataset.SyntheticDataset(
        4, 4, 100, 2, self._tile_images, self._tile_labels, label_fn=np.mean)
    expected_labels = {1, 5.5, 10, 50.5, 55, 100}
    self.assertEqual(expected_labels, set(ds.labels.flat))

    # Check that it works with more than 2 tiles_per_image.
    ds = synthetic_dataset.SyntheticDataset(
        4, 4, 100, 3, self._tile_images, self._tile_labels, label_fn=np.sum)
    triples = itertools.combinations_with_replacement(self._tile_labels, 3)
    expected_labels = set(map(sum, triples))
    self.assertEqual(expected_labels, set(ds.labels.flat))


class MnistSyntheticDatasetTest(unittest.TestCase):
  pass


class TileImageTest(unittest.TestCase):

  def setUp(self):
    self._tiles = [
      # C x H x W
      np.full((1, 2, 1), 1),
      np.full((1, 1, 3), 2),
      np.full((1, 2, 1), 0),
    ]

  def test_tile_image(self):
    tile_xys = [(1, 2), (0, 1), (0, 0)]

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

  def test_tile_image_merge_mode_max(self):
    tile_xys = [(1, 1), (0, 1), (0, 0)]

    expected = np.array(
        [[
          [0, 0, 0],
          [2, 2, 2],
          [0, 1, 0],
          [0, 0, 0],
        ]]
    )
    actual = synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4)
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_merge_mode_sum(self):
    tile_xys = [(1, 1), (0, 1), (0, 0)]

    expected = np.array(
        [[
          [0, 0, 0],
          [2, 3, 2],
          [0, 1, 0],
          [0, 0, 0],
        ]]
    )
    actual = synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4,
                                          merge_mode='sum')
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_merge_mode_fifo(self):
    tile_xys = [(1, 1), (0, 1), (0, 0)]

    expected = np.array(
        [[
          [0, 0, 0],
          [0, 2, 2],
          [0, 1, 0],
          [0, 0, 0],
        ]]
    )
    actual = synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4,
                                          merge_mode='fifo')
    np.testing.assert_array_equal(expected, actual)

  def test_tile_image_out_of_bounds(self):
    tile_xys = [(1, 2), (1, 1), (0, 0)]

    with self.assertRaises(ValueError):
      synthetic_dataset.tile_image(self._tiles, tile_xys, 3, 4)
