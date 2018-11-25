import collections
import os

from matplotlib import pyplot as plt
import numpy as np
import tensorboardX as tb

from rl.core.algs import util

Point = collections.namedtuple('Point', 'x y')

class Plotter(object):

  def __init__(self, log_dir=None):
    self.points = collections.defaultdict(list)
    self.figures = collections.defaultdict(plt.figure)
    if log_dir:
      log_dir = util.get_next_filename(log_dir)
      os.makedirs(log_dir)
    self.writer = SummaryWriter(log_dir)

  def add_data(self, key, x, y):
    self.points[key].append(Point(x, y))
    self.writer.add_scalar(key, np.mean(y), x)

  def get_data(self, key, x):
    points_list = self.points[key]
    for p in points_list:
      if p.x == x:
        return p.y
    return None

  def get_fig(self, key):
    return self.figures[key]

  def line_plot(self):
    for name, point_list in self.points.items():
      plt.figure(self.figures[name].number)
      xs = [p.x for p in point_list]
      try:
        iter(point_list[0].y)  # Checks if the provided ys are iterable.
        ys = np.array([np.mean(p.y) for p in point_list])
        std = np.array([np.std(p.y) for p in point_list])
        plt.fill_between(xs, ys - std, ys + std, alpha=0.2, label=name + ' std')
      except TypeError:
        ys = [p.y for p in point_list]
      plt.plot(xs, ys, label=name + ' mean')
      plt.title(name)
      plt.legend()

  def render(self):
    plt.pause(0.01)

  def get_means(self, points):
    try:
      iter(points[0].y)
      return np.array([np.mean(p.y) for p in points])
    except TypeError:
      return np.array([p.y for p in points])

  def top_k(self, key, k=1):
    """Returns the highest (x, y)s for 'key'.

    Args:
      key: metric name (a string).
      k: maximum number of top elements to return.

    Returns:
      List of up to |k| (x, y)s in descending order.
    """
    points_list = self.points[key]
    xs = np.array([p.x for p in points_list])
    ys = self.get_means(points_list)
    idx = np.argpartition(ys, -k)[-k:]
    idx = idx[np.argsort(ys[idx])][::-1]
    return zip(xs[idx], ys[idx])


class SummaryWriter(tb.SummaryWriter):
  def add_graph(self, model, input_to_model=None, **kwargs):
    if input_to_model is None:
      input_to_model = util.to_variable(np.zeros((1, 1)))
    return super().add_graph(model, input_to_model, **kwargs)

