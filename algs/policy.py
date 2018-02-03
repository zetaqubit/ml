"""Variety of policies and strategies for training them."""

import numpy as np
import torch as th

from rl.algs import util


class ImitationPolicy:
  """Continuous action policy, trained on imitation."""
  def __init__(self, model, lr=0.001):
    self.model = model
    self.optimizer = th.optim.Adam(self.model.parameters(), lr)

  def get_action(self, obs_np):
    return self.model.get_action(obs_np)

  def update(self, obs_batch, acs_batch):
    metrics = {}
    obs_var = util.to_variable(obs_batch)
    acs_var = util.to_variable(acs_batch)
    log_probs = self.model.log_probs(obs_var, acs_var, metrics)
    loss = -log_probs.mean()

    metrics['loss'] = util.to_numpy(loss)[0]

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics


class PolicyGradient(object):
  """Policy Gradient policy that learns from batches of episodes.

  Handles discrete and continuous action spaces.
  """
  def __init__(self, model, lr=0.005):
    self.model = model
    self.optimizer = th.optim.Adam(self.model.parameters(), lr)

  def get_action(self, obs_np):
    return self.model.get_action(obs_np)

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
    qs_var = util.to_variable(qs_batch)
    qs_var = (qs_var - qs_var.mean()) / qs_var.std()

    # Compute log-prob of the chosen actions under the current policy.
    acs_batch = np.array([sar.a for eps in eps_batch for sar in eps])
    obs_batch = np.array([sar.s for eps in eps_batch for sar in eps])
    acs_var = util.to_variable(acs_batch, dtype=None)
    obs_var = util.to_variable(obs_batch)
    log_probs = self.model.log_probs(obs_var, acs_var, metrics)

    # Scale by cumulative future rewards.
    log_p_q = log_probs * qs_var

    # Compute loss by negating the goal function J.
    j = log_p_q.sum() / len(eps_batch)
    loss = -j

    metrics['loss'] = util.to_numpy(loss)[0]

    # Compute gradients.
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics


