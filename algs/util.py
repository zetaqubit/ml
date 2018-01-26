import glob
import os
import re


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
