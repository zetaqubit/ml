"""Tests for rl.core.algs.util."""


import unittest

import numpy as np
import torch as th

from rl.core.algs import util


class UtilTest(unittest.TestCase):
  def setUp(self):
    self.model = th.nn.Sequential(
        th.nn.Conv2d(1, 4, 5),
        th.nn.ReLU(),
        th.nn.Conv2d(4, 8, 5),
        th.nn.ReLU(),
    )

  def test_num_params(self):
    expected = (
        1 * 4 * 5 * 5 +  # Layer 0 Conv2d
        4 +              # Layer 0 Bias
        4 * 8 * 5 * 5 +  # Layer 2 Conv2d
        8                # Layer 2 Bias
    )
    actual = util.num_params(self.model)
    self.assertEqual(expected, actual)

  def test_serialize_params(self):
    actual = util.serialize_params(self.model)
    self.assertEqual((util.num_params(self.model),), actual.shape)

    np.testing.assert_array_equal(
        self.model.state_dict()['0.weight'].numpy(),
        np.reshape(actual[:100], (4, 1, 5, 5)))

    np.testing.assert_array_equal(
        self.model.state_dict()['2.bias'].numpy(),
        actual[-8:])

  def test_deserialize_params(self):
    flattened = np.arange(util.num_params(self.model))
    util.deserialize_params(self.model, flattened)

    np.testing.assert_array_equal(
        np.reshape(flattened[:100], (4, 1, 5, 5)),
        self.model.state_dict()['0.weight'].numpy())

    np.testing.assert_array_equal(
        flattened[-8:],
        self.model.state_dict()['2.bias'].numpy())
