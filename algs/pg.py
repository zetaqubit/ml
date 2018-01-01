import collections
from pprint import pprint

import gym
import numpy as np
import torch as th
from torch import autograd as tha

dtype = th.cuda.FloatTensor

SAR = collections.namedtuple('SAR', 's a r')


class Environment(object):
  def __init__(self, env_name='CartPole-v0', max_episode_steps=None):
    self.env = gym.make(env_name)

    # Observation and action sizes
    discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
    assert discrete

    self.ob_dim = self.env.observation_space.shape[0]
    self.ac_dim = self.env.action_space.n
    self.max_episode_steps = max_episode_steps or self.env.spec.max_episode_steps
    self.env._max_episode_steps = self.max_episode_steps

  def sample_rollouts(self, policy, batch_size, render=False):
    """Samples complete episodes of at least |batch_size| under |policy|.

    Each episode is of at most self.max_episode_steps.
    Complete episodes (up to max_episode_steps) are sampled until at least
    |batch_size|.
    """
    episodes = []
    env_need_reset = True
    steps = 0
    while True:
      if env_need_reset:
        ob = self.env.reset()
        if steps >= batch_size:
          break
        episode = []
        episodes.append(episode)
      if render:
        self.env.render()
      ac = policy(ob)[0]
      ob, r, env_need_reset, _ = self.env.step(ac)
      episode.append(SAR(ob, ac, r))
      steps += 1
      env_need_reset |= len(episode) >= self.max_episode_steps
    return episodes


class Policy(object):
  def __init__(self, obs_dim, action_dim, lr=0.005):
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.model = self._create_policy_nn()
    self.optimizer = th.optim.Adam(self.model.parameters(), lr)

  def _create_policy_nn(self):
    hidden_dim = 64
    model = th.nn.Sequential(
      th.nn.Linear(self.obs_dim, hidden_dim),
      th.nn.Linear(hidden_dim, self.action_dim),
      th.nn.LogSoftmax(dim=1),
    )
    model.cuda()
    return model

  def get_action(self, obs_np):
    obs_var = tha.Variable(th.Tensor(obs_np).type(dtype))
    log_probs = self.model(obs_var.unsqueeze(0)).data
    probs = th.exp(log_probs).squeeze()
    ac = th.multinomial(probs, 1)
    out_np = ac.cpu().numpy()
    return out_np

  def update(self, eps_batch, discount=0.9):
    metrics = {'r_per_eps': []}

    # Compute cumulative discounted reward for each episode.
    qs_batch = []
    for eps in eps_batch:
      qs = np.array([sar.r for sar in eps])
      metrics['r_per_eps'].append(np.sum(qs))
      for t in range(len(eps) - 1, 0, -1):
        qs[t - 1] += discount * qs[t]
      qs_batch.append(qs)
    qs_batch = np.concatenate(qs_batch)
    qs_var = tha.Variable(th.Tensor(qs_batch).type(dtype))
    qs_var = (qs_var - qs_var.mean()) / qs_var.std()

    # Compute log-prob of the chosen actions under the current policy.
    acs_batch = np.array([sar.a for eps in eps_batch for sar in eps])
    obs_batch = np.array([sar.s for eps in eps_batch for sar in eps])
    acs_var = tha.Variable(th.from_numpy(acs_batch).unsqueeze(1).cuda())
    obs_var = tha.Variable(th.Tensor(obs_batch).type(dtype))
    log_probs = self.model(obs_var)
    log_probs = th.gather(log_probs, dim=1, index=acs_var)
    log_probs = log_probs.squeeze()

    # Scale by cumulative future rewards.
    log_p_q = log_probs * qs_var

    # Compute loss by negating the goal function J.
    j = log_p_q.sum() / len(eps_batch)
    loss = -j

    # Compute gradients.
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics

