import numpy as np

class Ewma(object):
  def __init__(self, alpha=0.99, save_every_n=100, log_every_n=None):
    self._alpha = alpha
    self._smoothed_val = None
    self._n = 0
    self._log_ever_n = log_every_n
    self._save_every_n = save_every_n
    self._history = []

  def add(self, new_val):
    if self._smoothed_val is None:
      self._smoothed_val = new_val
    else:
      self._smoothed_val = (
        self._alpha * self._smoothed_val + (1 - self._alpha) * new_val)

    self._n += 1
    if self._n % self._save_every_n == 0:
      self._history.append(self._smoothed_val)
    if self._log_ever_n and self._n % self._log_ever_n == 0:
      print(self._n, 'smoothed:', self._smoothed_val)


  def get(self):
    return self._smoothed_val

  def history(self):
    indices = np.arange(0, self._save_every_n * len(self._history),
                        self._save_every_n)
    return np.vstack((indices, np.array(self._history)))


