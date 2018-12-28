"""A synthetic dataset where each image is composed of multiple MNIST digits.

Large images are synthesized by placing several MNIST digits at different
locations on top of a blank background.

The label is a function of the labels of MNIST digits chosen; for example,
it could be the sum, mean, or max of the digits present in the synthesized
image.

This is intended to be a generalization of the Cluttered Translated MNIST
task [1].

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
"""

import numpy as np
import torch as th
import torchvision as tv


class SyntheticDataset(th.utils.data.TensorDataset):

  def __init__(self, width, height, num_images, tiles_per_image,
               tile_images, tile_labels, label_fn=np.sum, seed=0):
    """

    :param width:
    :param height:
    :param num_images:
    :param tiles_per_image:
    :param tile_images: shape [n, c, h, w]
    :param tile_labels: shape [n]
    :param label_fn: (tile_label1, tile_label2, ...) -> image_label
    :param seed:
    """
    assert tile_images.ndim == 4
    assert tile_labels.ndim == 1
    assert tile_images.shape[0] == tile_labels.shape[0] > 0

    self._width = width
    self._height = height
    self._tiles_per_image = tiles_per_image

    self._rand = np.random.RandomState(seed)

    self._images = np.empty((num_images, 1, height, width))
    self._labels = np.empty((num_images,))

    for i in range(num_images):
      self._images[i], self._labels[i] = self._construct_example(
          tile_images, tile_labels, label_fn)

    super().__init__(th.Tensor(self._images), th.Tensor(self._labels))

  def _construct_example(self, tile_images, tile_labels, label_fn):
    tile_n, tile_c, tile_h, tile_w = tile_images.shape
    select_idxs = self._rand.choice(tile_n, size=self._tiles_per_image)
    select_images = tile_images[select_idxs]
    select_labels = tile_labels[select_idxs]

    # Sample (x, y) for each tile.
    # TODO: add option to disallow overlapping tile placement.
    max_x = self._width - tile_w
    max_y = self._height - tile_h
    tile_xys = [(self._rand.randint(max_x), self._rand.randint(max_y))
                 for _ in range(self._tiles_per_image)]

    image = tile_image(select_images, tile_xys, self._width, self._height)
    label = label_fn(select_labels)
    return image, label

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels


def MnistSyntheticDataset(mnist_dir='~/code/data/mnist', train=True, **kwargs):
  mnist_dataset = tv.datasets.MNIST(
      mnist_dir, train, transform=tv.transforms.ToTensor(), download=True)
  images = mnist_dataset.data.numpy()
  labels = mnist_dataset.targets.numpy()

  # [n, h, w] -> [n, c, h, w]
  images = np.expand_dims(images, axis=1)

  return SyntheticDataset(tile_images=images, tile_labels=labels, **kwargs)


def tile_image(tiles, tile_xys, width, height):
  """Creates an image by blitting tiles at specified locations.

  :param tiles: list of np.arrays shaped [c, h, w]
  :param tile_xys: corresponding list of locations to place the tiles. These
    are the output coordinates of the top-left corners of the tiles.
    Format: [(x_1, y_1), (x_2, y_2), ...]
  :param width: width of the output image, in pixels
  :param height: height of the output image, in pixels
  :return: the merged image, as an np.array shaped [c, height, width]
  """
  channels = tiles[0].shape[0]
  image = np.zeros((channels, height, width))
  for tile, (x, y) in zip(tiles, tile_xys):
    tile_c, tile_h, tile_w = tile.shape
    image[:, y:y + tile_h, x:x + tile_w] = tile
  return image
