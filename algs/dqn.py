import copy
from pprint import pprint
from typing import List

import numpy as np
import torch as th
import torch.nn.functional as thf

from rl.algs import environment
from rl.algs import util

Sars = environment.SARS


class ReplayBuffer:
  def __init__(self, size=10000):
    self.max_size = size
    self.buf = [None] * self.max_size
    self.i = 0
    self.size = 0

  def add(self, entry):
    self.buf[self.i] = entry
    self.i = (self.i + 1) % self.max_size
    if self.size < self.max_size:
      self.size += 1

  def sample(self, n=1):
    """Samples n entries uniformly at random, without replacement."""
    assert n <= self.size
    idxs = np.random.choice(self.size, n, replace=False)
    entries = [self.buf[i] for i in idxs]
    return entries

  def all_entries(self):
    return self.buf[:self.size]


class Dqn:
  def __init__(self, model, lr, eps_sched, target_update_freq=10000):
    self.model = model
    self.target_model = copy.deepcopy(self.model)
    self.optimizer = th.optim.Adam(self.model.parameters(), lr)
    self.eps_sched = eps_sched
    self.target_update_freq = target_update_freq
    self.gamma = 0.99
    self.step = 0

  def action_values(self, obs_np):
    obs_var = util.to_variable(obs_np)
    return self.model(obs_var)

  def get_action(self, obs_np):
    qs = self.action_values(obs_np)
    probs = thf.softmax(qs, dim=-1)
    probs = util.to_numpy(probs)
    return util.sample_eps_greedy(probs, self.eps_sched.get(self.step))

  def _update_step_params(self):
    self.step += 1
    if self.step % self.target_update_freq == 0:
      # print(f'step {self.step}: updating target model')
      # print(f'eps: {self.eps_sched.get(self.step)}')
      self.target_model = copy.deepcopy(self.model)

  def update(self, sars_batch: List[Sars]):
    metrics = {}
    self._update_step_params()
    bs = len(sars_batch)
    empty = np.zeros_like(sars_batch[0].s)
    s_batch = np.stack([sars.s for sars in sars_batch])
    a_batch = np.stack([sars.a for sars in sars_batch])
    r_batch = util.to_variable(np.stack([sars.r for sars in sars_batch]))
    s1_batch = np.stack([sars.s1 if sars.s1 is not None else empty
                         for sars in sars_batch])

    #pprint(list(zip(s_batch.squeeze(), s1_batch.squeeze())))
    #print(a_batch.squeeze())
    #print(r_batch.squeeze())

    terminal_mask = util.to_variable(
      np.stack([1 if sars.s1 is None else 0 for sars in sars_batch]))

    a_var = util.to_variable(a_batch, dtype=th.LongTensor).unsqueeze(dim=1)

    qs = self.action_values(s_batch)
    qs_sel = th.gather(qs, dim=-1, index=a_var)

    s1_var = util.to_variable(s1_batch, volatile=True)
    target_qs = self.target_model(s1_var)
    target_qs_max, qs_max_idx = th.max(target_qs, dim=-1)
    target = self.gamma * terminal_mask * target_qs_max
    target += r_batch

    loss = thf.mse_loss(qs_sel, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    metrics['loss'] = util.to_numpy(loss)[0]
    metrics['epsilon'] = self.eps_sched.get(self.step)
    return metrics

