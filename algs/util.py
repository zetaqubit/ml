import glob
import os
import re

import numpy as np
import torch as th
from torch import autograd as tha


def to_variable(x, dtype=th.FloatTensor):
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
  return tha.Variable(x).cuda()


def to_numpy(x):
  """Converts torch.Tensor or tha.Variable to nmupy array."""
  if isinstance(x, tha.Variable):
    x = x.data
  return x.cpu().numpy()


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
