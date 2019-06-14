"""Event logging for Box2D simulation.
"""
import collections
import pandas as pd

from rl.box2d.newton import box2d_state
from rl.core.algs import util


class CsvLog:
  def __init__(self, log_filepath=None, log_dir=None):
    if not log_filepath:
      log_filepath = util.get_next_filename(log_dir, extension='.csv')
    self.filepath = log_filepath
    self._rows = []

  def add(self, key_values):
    self._rows.append(collections.OrderedDict(key_values))

  def as_df(self):
    return pd.DataFrame(self._rows)

  def write(self):
    df = self.as_df()
    df.to_csv(self.filepath, index=False)

  def num_events(self):
    return len(self._rows)


class Box2DLog(CsvLog):
  def __init__(self, world, **kwargs):
    super().__init__(**kwargs)
    self._world = world

  def add_world_state(self, step):
    state = {'step': step}
    state.update(box2d_state.dynamic_world_state(self._world))
    self.add(state)



