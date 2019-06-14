"""Tests for rl.box2d.newton.box2d_log."""

import io

import pandas as pd

from rl.box2d.newton import box2d_log


def test_add(tmp_path):
  log = box2d_log.CsvLog(log_filepath=tmp_path)
  row = {'t': 1, 'a': 2, 'b': 3}
  log.add(row)
  expected = pd.DataFrame([row], columns=['t', 'a', 'b'])
  pd.testing.assert_frame_equal(expected, log.as_df())

def test_add_different_keys(tmpdir):
  log = box2d_log.CsvLog(log_dir=tmpdir)
  rows = [
      {'t': 1, 'a': 2, 'b': 3},
      {'t': 2, 'c': 4, 'b': 5},
  ]
  log.add(rows[0])
  log.add(rows[1])
  expected = pd.DataFrame(rows, columns=['t', 'a', 'b', 'c'])
  pd.testing.assert_frame_equal(expected, log.as_df())

def test_write():
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

