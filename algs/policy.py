"""Variety of policies and strategies for training them."""

import numpy as np
import torch as th
from torch.nn import functional as thf

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

    metrics['loss'] = util.to_numpy(loss).item()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics


class PolicyGradient(object):
  """Policy Gradient policy that learns from batches of episodes.

  Handles discrete and continuous action spaces.
  """
  def __init__(self, model, value_nn=None, lr=0.005, discount=1.0):
    self.discount = discount
    self.model = model
    self.value_nn = value_nn
    parameters = list(model.parameters())
    if value_nn:
      parameters.extend(value_nn.parameters())
    self.optimizer = th.optim.Adam(parameters, lr)

  def get_action(self, obs_np):
    return self.model.get_action(obs_np)

  def _compute_qs(self, eps_batch, metrics=None):
    """Computes cumulative discounted rewards."""
    if metrics is not None:
      metrics['r_per_eps'] = []
    qs_batch = []
    for eps in eps_batch:
      qs = np.array([sar.r for sar in eps])
      if metrics is not None:
        metrics['r_per_eps'].append(np.sum(qs))
      for t in range(len(eps) - 1, 0, -1):
        qs[t - 1] += self.discount * qs[t]
      qs_batch.append(qs)
    qs_batch = np.concatenate(qs_batch)
    return util.to_variable(qs_batch)

  def _compute_advantages(self, obs, qs, metrics=None):
    """Computes the advantage."""
    qs_normed = util.normalize(qs)
    if not self.value_nn:
      return qs_normed, util.to_variable(np.array([0]))

    vs = self.value_nn(obs).squeeze()
    qs_mean, qs_std = th.mean(qs), th.std(qs)
    advs_normed = qs_normed - vs

    # Optimize the Value NN to predict normalized Vs.
    loss = thf.mse_loss(vs, qs_normed)
    #loss = thf.smooth_l1_loss(vs, qs_normed)

    if metrics is not None:
      metrics['qs_mean'] = util.to_numpy(qs_mean)
      agree = (qs_normed * vs) >= 0
      metrics['q_v_agree'] = util.to_numpy(th.sum(agree)) / qs.shape[0]

    output = advs_normed
    output = output.detach()
    return output, loss

  def update(self, eps_batch):
    metrics = {}

    acs_batch = np.array([sar.a for eps in eps_batch for sar in eps])
    obs_batch = np.array([sar.s for eps in eps_batch for sar in eps])
    acs_var = util.to_variable(acs_batch, dtype=None)
    obs_var = util.to_variable(obs_batch)

    # Compute normalized discounted rewards.
    qs = self._compute_qs(eps_batch, metrics)
    advs, value_loss = self._compute_advantages(obs_var, qs, metrics)
    metrics['advs'] = util.to_numpy(advs)[::20]

    # Compute log-prob of the chosen actions under the current policy, scaled
    # by advantages
    log_probs = self.model.log_probs(obs_var, acs_var, metrics)
    log_p_q = log_probs * advs

    # Compute loss by negating the goal function J.
    j = log_p_q.sum(dim=0) / len(eps_batch)
    policy_loss = -j
    loss = policy_loss + value_loss

    metrics['loss/total'] = util.to_numpy(loss).item()
    metrics['loss/policy'] = util.to_numpy(policy_loss).item()
    metrics['loss/value'] = util.to_numpy(value_loss).item()

    # Compute gradients.
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics
