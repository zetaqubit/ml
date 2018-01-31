import copy

import numpy as np

from rl.algs import plotter
from rl.algs.environment import Environment


class Experiment:
  def __init__(self, env_name, model_cls, model_dict, policy_cls, policy_dict):
    self.env = Environment(env_name=env_name)
    self.plt = plotter.Plotter()
    self.snapshots = Snapshots()

    model_dict = dict(model_dict)
    model_dict.update({
      'obs_dim': self.env.obs_dim,
      'acs_dim': self.env.acs_dim,
    })
    model = model_cls(**model_dict)
    self.policy = policy_cls(model, **policy_dict)

  def visualize(self, step=None, top_k=1, num_episodes=1):
    """Runs the policy from either the best steps or at specified step.

    Defaults to running 1 episode with the best policy encountered during
    training.
    """
    if step is None:
      steps_rewards = self.plt.top_k('r_per_eps', top_k)
    else:
      rewards = [np.mean(self.plt.get_data('r_per_eps', step))]
      steps_rewards = zip([step], rewards)

    for step, reward in steps_rewards:
      print(f'Step: {step}. Expected reward: {reward}')
      rs = self.env.visualize(self.snapshots.get(step),
                              num_episodes=num_episodes)
      print(f'Actual rewards: [{rs}]')


class Snapshots(object):
  def __init__(self):
    self._snapshots = {}

  def add(self, episode, policy):
    self._snapshots[episode] = copy.deepcopy(policy)

  def get(self, episode):
    return self._snapshots[episode]
