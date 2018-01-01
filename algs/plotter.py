from matplotlib import pyplot as plt

class Plotter(object):
  def __init__(self):
    self.xs = []
    self.ys = []

  def add_data(self, x, y):
    self.xs.append(x)
    self.ys.append(y)

  def create_fig(self):
    return plt.figure()

  def line_plot(self, figure=None):
    fig = figure or plt.figure()
    plt.plot(self.xs, self.ys)
    return fig

  def render(self):
    plt.pause(0.01)
