import glob
import os
import re
from typing import List

import numpy as np
import torch as th
from torch import autograd as tha


def to_variable(x, dtype=th.FloatTensor, **kwargs):
  """Converts numpy array or torch.Tensor to a Variable.

  Args:
    x: np.ndarray, or torch.Tensor
    dtype: variable datatype. Will also be converted to cuda.
      If none, will be inferred from the nd_array.
  """
  if isinstance(x, np.ndarray):
    x = th.from_numpy(x)
  if dtype is not None:
    x = x.type(dtype)
  return tha.Variable(x, **kwargs).cuda()


def to_numpy(x):
  """Converts torch.Tensor or tha.Variable to nmupy array."""
  if isinstance(x, tha.Variable):
    x = x.data
  return x.cpu().numpy()


def normalize(x, to_mean=0.0, to_std=1.0, eps=1e-8):
  """Normalizes array to have specified mean and std."""
  from_mean, from_std = x.mean(), x.std()
  x = (x - from_mean) / (from_std + eps)
  return x * (to_std + eps) + to_mean


def print_weights(model):
  for m in model.modules():
    if isinstance(m, th.nn.Linear):
      print(f'{m}: w: {m.weight.data}')
      print(f'{m}: b: {m.bias.data}')


def get_next_filename(dir_path, prefix='', extension=''):
  """Gets the next untaken file name in a directory.

  The files are of form <prefix>1, <prefix>2, ...
  """
  files = glob.glob(os.path.join(dir_path, prefix + '*' + extension))
  numerals = []
  for file in files:
    name = os.path.basename(file)
    pattern = prefix + r'(\d+)' + extension
    m = re.match(pattern, name)
    if m:
      numerals.append(int(m.group(1)))
  if len(numerals) == 0:
    return os.path.join(dir_path, prefix + '1' + extension)

  numerals = sorted(numerals)
  return os.path.join(dir_path, prefix + str(numerals[-1] + 1) + extension)


def sample_eps_greedy(probs, eps=0):
  """Samples the distribution under epsilon-greedy."""
  n = probs.shape[-1]

  def sample(prob):
    if np.random.rand() < eps:
      return np.random.choice(n)
    else:
      return np.argmax(prob)

  dims = len(probs.shape)
  if dims == 1:
    return sample(probs)
  elif dims == 2:
    num_rows = probs.shape[0]
    selected = np.empty(num_rows)
    for i in range(num_rows):
      selected[i] = sample(probs[i, :])
    return selected


class Schedule:
  """Piecewise linear function."""
  def __init__(self, xs: List[int], ys: List[int]):
    if not np.all(np.diff(xs) >= 0):
      raise ValueError('x coordinates must be sorted in ascending order.')
    self.xs = xs
    self.ys = ys

  def get(self, x):
    return np.interp(x, self.xs, self.ys)
