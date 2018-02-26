"""Wrapper around gym.Environment that supports sampling rollouts of a policy.
"""
import collections

import gym
import numpy as np


SAR = collections.namedtuple('SAR', 's a r')


class Environment(object):
  def __init__(self, env_name='CartPole-v0', max_episode_steps=None):
    self.env_name = env_name
    self.max_episode_steps = max_episode_steps
    self.env = None
    self.reset()

    # Observation and action sizes
    self.discrete_ac = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.obs_dim = self.env.observation_space.shape[0]
    self.acs_dim = (self.env.action_space.n if self.discrete_ac
                    else self.env.action_space.shape[0])

  def reset(self):
    self.env = gym.make(self.env_name)
    if self.max_episode_steps:
      self.env._max_episode_steps = self.max_episode_steps
    else:
      self.max_episode_steps = self.env.spec.max_episode_steps

  def sample_rollouts(self, policy, batch_size=None, num_episodes=None,
                      render=False):
    """Samples complete episodes of at least |batch_size| under |policy|.

    Each episode is of at most self.max_episode_steps.
    If batch_size is specified, complete episodes (up to max_episode_steps) are
      sampled until at least |batch_size| steps.
    If num_episodes is specified, |num_episodes| full episodes are sampled.
    """
    assert batch_size or num_episodes

    episodes = []
    env_need_reset = True
    episode_i = 0
    steps = 0
    while True:
      if env_need_reset:
        ob = self.env.reset()
        if batch_size and steps >= batch_size:
          break
        if num_episodes and episode_i >= num_episodes:
          break
        episode_i += 1
        episode = []
        episodes.append(episode)
      if render:
        self.env.render()
      ac = policy(ob)
      ob, r, env_need_reset, _ = self.env.step(ac)
      episode.append(SAR(ob, ac, r))
      steps += 1
      env_need_reset |= len(episode) >= self.max_episode_steps
    return episodes

  def visualize(self, policy, num_episodes=1):
    try:
      eps = self.sample_rollouts(policy.get_action, num_episodes=num_episodes,
                                 render=True)
      rs = [sum([sar.r for sar in ep]) for ep in eps]
      print(f'Reward: mean {np.mean(rs):.2f}, std {np.std(rs):.2f} '
            f'over {num_episodes} episodes.')
      return rs
    finally:
      self.env.close()
      self.reset()


