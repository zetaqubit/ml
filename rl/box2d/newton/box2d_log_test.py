"""Tests for rl.box2d.newton.box2d_log."""

import io
import unittest

import pandas as pd

from rl.box2d.newton import box2d_log


class CsvLogTest(unittest.TestCase):
  def setup(self):
    pass

  def test_add(self):
    log = box2d_log.CsvLog(None)
    row = {'t': 1, 'a': 2, 'b': 3}
    log.add(row)
    expected = pd.DataFrame([row])
    pd.testing.assert_frame_equal(expected, log.as_df())

  def test_add_different_keys(self):
    log = box2d_log.CsvLog(None)
    rows = [
        {'t': 1, 'a': 2, 'b': 3},
        {'t': 2, 'c': 4, 'b': 5},
    ]
    log.add(rows[0])
    log.add(rows[1])
    expected = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(expected, log.as_df())

  def test_write(self):
    fp = io.StringIO()
    log = box2d_log.CsvLog(fp)
    rows = [
        {'t': 1, 'a': 2, 'b': 3},
        {'t': 2, 'c': 4, 'b': 5},
    ]
    log.add(rows[0])
    log.add(rows[1])
    log.write()

    fp.seek(0)
    actual = pd.read_csv(fp)
    pd.testing.assert_frame_equal(log.as_df(), actual)

