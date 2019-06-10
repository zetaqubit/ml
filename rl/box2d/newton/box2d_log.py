"""Event logging for Box2D simulation.
"""
import pandas as pd


class CsvLog:
  def __init__(self, log_filepath):
    self._filepath = log_filepath
    self._rows = []

  def add(self, key_values):
    self._rows.append(key_values)

  def as_df(self):
    return pd.DataFrame(self._rows)

  def write(self):
    df = self.as_df()
    df.to_csv(self._filepath, index=False)


class Box2DLog(CsvLog):
  def __init__(self, log_filepath):
    self.super().__init__(log_filepath)

