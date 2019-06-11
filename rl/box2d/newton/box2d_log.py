"""Event logging for Box2D simulation.
"""
import collections
import pandas as pd

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


def object_state(b2_obj, name):
  state = {}
  pos = b2_obj.worldCenter
  state[name + '.pos_x'] = pos[0]
  state[name + '.pos_y'] = pos[1]
  state[name + '.pos_a'] = b2_obj.angle

  vel = b2_obj.linearVelocity
  state[name + '.vel_x'] = vel[0]
  state[name + '.vel_y'] = vel[1]
  state[name + '.vel_a'] = b2_obj.angularVelocity
  return state


