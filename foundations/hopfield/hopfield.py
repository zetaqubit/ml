import dataclasses

import numpy as np


class HopfieldNetwork:
  """Hopfield network for storing and retrieving patterns.

  States binary, stored as 0 and 1.
  """

  def __init__(self, d: int):
    self.d = d
    self.state = np.zeros(d)
    # TODO: random init to break symmetry?
    self.weights = np.zeros((d, d), dtype=np.float32)
    self.biases = np.zeros(d)

  def fit(self, data: np.ndarray):
    """Fits the Hopfield network to the provided data.

    Args:
      data: patterns to memorize. Shape n x d.
    """
    assert len(data.shape) == 2
    for pattern in data:
      a = 2 * (pattern - 0.5)
      outer_product = np.outer(a, a)
      np.fill_diagonal(outer_product, 0)
      self.weights += outer_product
      self.biases += a

  def predict(self, pattern):
    self.state = pattern[:]
    for i in range(10):
      self.iterate()
    return self.state

  def iterate(self):
    """Iterates self.state to a low-energy configuration."""
    node_order = np.random.permutation(self.d)
    for node_idx in node_order:
      node_energy = -np.inner(self.weights[node_idx, :], self.state)
      node_energy -= self.biases[node_idx]
      self.state[node_idx] = 1 if node_energy < 0 else 0


if __name__ == '__main__':
  net = HopfieldNetwork(10)

  net.fit(np.array([
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  ], dtype=np.float32))

  print(net.predict([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
  print(net.predict([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))

