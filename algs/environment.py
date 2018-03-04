"""Wrapper around gym.Environment that supports sampling rollouts of a policy.
"""
import collections

import gym
import numpy as np
from skimage import color
from skimage import transform


SAR = collections.namedtuple('SAR', 's a r')
SARS = collections.namedtuple('SARS', 's a r s1')


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
        if batch_size and steps >= batch_size:
          break
        if num_episodes and episode_i >= num_episodes:
          break
        episode_i += 1
        episode = []
        episodes.append(episode)
        ob = self.env.reset()
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


class AtariEnvironment:
  def __init__(self, env_name, action_repeat=4):
    self.env_name = env_name + 'NoFrameskip-v4'
    self.action_repeat = action_repeat
    self.env = gym.make(self.env_name)

    # Observation and action sizes
    self.discrete_ac = isinstance(self.env.action_space, gym.spaces.Discrete)
    self.raw_obs_shape = self.env.observation_space.shape
    self.obs_dim = (action_repeat, 84, 84)
    self.acs_dim = (self.env.action_space.n if self.discrete_ac
                    else self.env.action_space.shape[0])

    self.last_obs = self.reset()

  def reset(self):
    obs = np.empty(self.obs_dim)
    obs[0, :, :] = self._preprocess(self.env.reset())
    # TODO: verify assumption that action 0 is No-op.
    obs[1:, :, :], r, done = self.step_k(0, self.action_repeat - 1)
    assert not done  # TODO: handle
    return obs

  def _preprocess(self, ob):
    """Converts 210x160x3 RGB to 84x84x1 Y."""
    y = color.rgb2yuv(ob)[:, :, 0]
    y_resized = transform.resize(y, (84, 84))
    return y_resized

  def step_k(self, ac, k):
    obs = np.zeros((k,) + self.obs_dim[1:])
    r_sum = 0
    done = False
    for i in range(k):
      rgb, r, done, _ = self.env.step(ac)
      obs[i, :, :] = self._preprocess(rgb)
      r_sum += r  # TODO: might need to discount
      if done:
        break
    return obs, r_sum, done

  def step(self, policy):
    assert self.last_obs is not None
    print(self.last_obs.shape)
    ac = policy(np.expand_dims(self.last_obs, 0)).squeeze().astype(int)
    obs, r, done = self.step_k(ac, self.action_repeat)
    if not done:
      sars = SARS(self.last_obs, ac, r, obs)
      self.last_obs = obs
    else:
      sars = SARS(self.last_obs, ac, r, None)
      self.last_obs = self.reset()
    return sars
