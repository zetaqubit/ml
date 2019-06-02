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
  """A synthetic image dataset created by sampling and composing tiles.

  Given a source image dataset of (tile_images, tile_labels) shaped
  ([n, c, h, w], [n]), construct the SyntheticDataset of
  (synth_images, synth_labels) shaped ([N, c, H, W]), [N]), where N, H, and W
  are user-specified.

  Each image is synthesized by sampling `tiles_per_image` tiles and placing each
  at a random location in the WxH output image, initially all 0s. The
  label is synthesized by applying a function (defaulting to sum) over the
  labels of the sampled tiles.
  """

  def __init__(self, width, height, num_images, tiles_per_image,
               tile_images, tile_labels, label_fn=np.sum, seed=0):
    """Creates a SyntheticDataset from a tile dataset.

    :param width: width of the synthesized image, in pixels
    :param height: height of the synthesized image, in pixels
    :param num_images: number of images to generate
    :param tiles_per_image: number of tiles to sample for each synthesized image
    :param tile_images: tiles to use to construct images. np.array shaped
      [n, c, h, w]
    :param tile_labels: labels of the tiles, used to construct the image labels.
      np.array, shape [n]
    :param label_fn: function that takes in `tiles_per_image` tile labels and
      outputs the image label. Defaults to sum of tile labels.
      np.array([tile_label1, tile_label2, ...]) -> image_label
    :param seed: random seed used in sampling tiles and positioning them in the
      synthesized images.
    """
    assert tile_images.ndim == 4
    assert tile_labels.ndim == 1
    assert tile_images.shape[0] == tile_labels.shape[0] > 0

    channels = tile_images.shape[1]
    self._width = width
    self._height = height
    self._tiles_per_image = tiles_per_image

    self._rand = np.random.RandomState(seed)

    self._images = np.empty((num_images, channels, height, width))
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
    max_x = self._width - tile_w + 1
    max_y = self._height - tile_h + 1
    tile_xys = [(self._rand.randint(max_x), self._rand.randint(max_y))
                for _ in range(self._tiles_per_image)]

    image = tile_image(select_images, tile_xys, self._width, self._height)
    label = label_fn(select_labels)
    return image, label

  @property
  def images(self):
    """Synthesized images, as np.array shaped [num_images, c, height, width]."""
    return self._images

  @property
  def labels(self):
    """Synthesized labels, as np.array shaped [num_images]."""
    return self._labels


class MnistSyntheticDataset(SyntheticDataset):
  """SyntheticDataset created from MNIST image tiles.

  Each synthesized image contains randomly-placed MNIST images. The synthesized
  label is a function (defaulting to sum) of the MNIST digit labels.
  """

  def __init__(self, width, height, num_images, tiles_per_image,
               mnist_dir='~/code/data/mnist', train=True, label_fn=np.sum,
               seed=0):
    """Creates an MnistSyntheticDataset using MNIST as a tile dataset.

    :param width: width of the synthesized image, in pixels
    :param height: height of the synthesized image, in pixels
    :param num_images: number of images to generate
    :param tiles_per_image: number of tiles to sample for each synthesized image
    :param mnist_dir: directory to load MNIST from if it exists; if not,
      downloads MNIST to this directory.
    :param train: whether to use the MNIST training or test set.
    :param label_fn: function that takes in `tiles_per_image` tile labels and
      outputs the image label. Defaults to sum of tile labels.
      np.array([tile_label1, tile_label2, ...]) -> image_label
    :param seed: random seed used in sampling tiles and positioning them in the
      synthesized images.
    """
    mnist_dataset = tv.datasets.MNIST(
        mnist_dir, train, transform=tv.transforms.ToTensor(), download=True)
    self.orig_images = mnist_dataset.data.numpy()
    self.orig_labels = mnist_dataset.targets.numpy()

    # [n, h, w] -> [n, c, h, w]
    self.orig_images = np.expand_dims(self.orig_images, axis=1)

    super().__init__(width, height, num_images, tiles_per_image,
                     tile_images=self.orig_images,
                     tile_labels=self.orig_labels,
                     label_fn=label_fn, seed=seed)


class ClutteredMnistSyntheticDataset(SyntheticDataset):
  def __init__(self, width, height, num_images, true_tiles_per_image,
               clutter_tiles_per_image, mnist_dir='~/code/data/mnist',
               train=True, label_fn=np.sum, seed=0, merge_mode='max'):

    # Create uncluttered dataset.
    true_ds = MnistSyntheticDataset(
        width, height, num_images, true_tiles_per_image,
        mnist_dir=mnist_dir, train=train, label_fn=label_fn, seed=seed)

    # Create clutter tiles by splitting each true tile into 2x2.
    clutter_tiles = np.concatenate(np.split(true_ds.orig_images, 2, axis=2))
    clutter_tiles = np.concatenate(np.split(clutter_tiles, 2, axis=3))
    clutter_labels = np.tile(true_ds.orig_labels, 4)
    clutter_ds = SyntheticDataset(
        width, height, num_images, clutter_tiles_per_image, clutter_tiles,
        clutter_labels)

    # Merge with clutter background.
    # TODO: fix fifo - it replaces entire image rather than just foreground.
    merge_fn = _get_merge_fn(merge_mode)
    self._images = merge_fn(true_ds.images, clutter_ds.images)
    self._labels = true_ds.labels

    th.utils.data.TensorDataset.__init__(self,
                                         th.Tensor(self._images),
                                         th.Tensor(self._labels))


def tile_image(tiles, tile_xys, width, height, merge_mode='max'):
  """Creates an image by blitting tiles at specified locations.

  :param tiles: list of np.arrays shaped [c, h, w]
  :param tile_xys: corresponding list of locations to place the tiles. These
    are the output coordinates of the top-left corners of the tiles.
    Format: [(x_1, y_1), (x_2, y_2), ...]
  :param width: width of the output image, in pixels
  :param height: height of the output image, in pixels
  :param merge_mode: how to merge two tiles if they overlap. The output pixel
    value depends on the mode:
      - 'max' (default): take the maximum pixel value over all overlapping tiles
      - 'sum': take the sum of pixel values over all overlapping tiles
      - 'fifo': tiles are painted in order. Take the pixel of the last tile.
  :return: the merged image, as an np.array shaped [c, height, width]
  """
  merge_fn = _get_merge_fn(merge_mode)

  channels = tiles[0].shape[0]
  image = np.zeros((channels, height, width))
  for tile, (x, y) in zip(tiles, tile_xys):
    tile_c, tile_h, tile_w = tile.shape
    old = image[:, y:y + tile_h, x:x + tile_w]
    image[:, y:y + tile_h, x:x + tile_w] = merge_fn(old, tile)
  return image


def _get_merge_fn(merge_mode):
  merge_fns = {
      'max': np.maximum,
      'sum': np.add,
      'fifo': lambda old, new: new,
  }
  assert merge_mode in merge_fns.keys()
  return merge_fns[merge_mode]
