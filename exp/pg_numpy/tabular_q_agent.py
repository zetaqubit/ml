from collections import defaultdict
import numpy as np

from gym.spaces import discrete

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


class TabularQAgent(object):
  """
  Agent implementing tabular Q-learning.
  """

  def __init__(self, observation_space, action_space, **userconfig):
    if not isinstance(observation_space, discrete.Discrete):
      raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
    if not isinstance(action_space, discrete.Discrete):
      raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
    self.observation_space = observation_space
    self.action_space = action_space
    self.action_n = action_space.n
    self.config = {
        "init_mean" : 0.0,      # Initialize Q values with this mean
        "init_std" : 0.0,       # Initialize Q values with this standard deviation
        "learning_rate" : 0.1,
        "eps": 0.05,            # Epsilon in epsilon greedy policies
        "discount": 0.95,
        "n_iter": 1000000}        # Number of iterations
    self.config.update(userconfig)
    self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

  def act(self, observation, eps=None):
    if eps is None:
      eps = self.config["eps"]
    # epsilon greedy.
    if np.random.random() > eps:
      # Of all actions with max Q, select one uniformly at random.
      q_max = np.max(self.q[observation])
      idx_max = np.arange(self.action_n)[self.q[observation] == q_max]
      action = np.random.choice(idx_max)
    else:
      action = self.action_space.sample()
    return action

  def learn(self, env):
    smoothed_reward = Ewma(log_every_n=1000)
    obs = env.reset()
    q = self.q
    num_iters = self.config['n_iter']
    episode_r = 0
    for t in range(num_iters):
      action = self.act(obs)
      obs2, reward, done, _ = env.step(action)
      episode_r += reward
      future = 0.0
      if not done:
        future = np.max(q[obs2])
      else:
        smoothed_reward.add(episode_r)
        episode_r = 0
        obs2 = env.reset()
      q[obs][action] -= \
          self.config["learning_rate"] * (q[obs][action] - reward - self.config["discount"] * future)
      obs = obs2
    return smoothed_reward.history()

