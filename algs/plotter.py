import collections
from matplotlib import pyplot as plt
import numpy as np

Point = collections.namedtuple('Point', 'x y')

class Plotter(object):

  def __init__(self):
    self.points = collections.defaultdict(list)
    self.figures = {}

  def add_data(self, key, x, y):
    self.points[key].append(Point(x, y))

  def create_figs(self, keys):
    self.figures.update({k: plt.figure() for k in keys})

  def get_fig(self, key):
    return self.figures[key]

  def line_plot(self):
    for name, point_list in self.points.items():
      plt.figure(self.figures[name].number)
      xs = [p.x for p in point_list]
      try:
        iter(point_list[0].y)  # Checks if the provided ys are iterable.
        ys = [np.mean(p.y) for p in point_list]
        ys_low = [np.percentile(p.y, 2.5) for p in point_list]
        ys_high = [np.percentile(p.y, 97.5) for p in point_list]
        plt.fill_between(xs, ys_low, ys_high, alpha=0.4, label=name + ' 95%')
      except TypeError:
        ys = [p.y for p in point_list]
      plt.plot(xs, ys, label=name + ' mean')
      plt.title(name)
      plt.legend()

  def render(self):
    plt.pause(0.01)
