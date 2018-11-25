"""PyTorch implementation of Behavior Cloning and DAgger algorithms.
"""

import collections
import os
import pickle
import tensorflow as tf

import numpy as np
import torch as th
from torch.utils import data

from rl.berkeleyrlcourse.hw1 import load_policy
from rl.berkeleyrlcourse.hw1 import tf_util
from rl.core.envs import environment
from rl.core.algs import experiment, policy, model


class ExpertDataset:
  def __init__(self, obs, acs, batch_size):
    self.reset(obs, acs, batch_size)

  @staticmethod
  def from_pkl(pkl_file, batch_size):
    with open(pkl_file, 'rb') as fd:
      rollouts = pickle.load(fd)
    obs = rollouts['observations']
    acs = rollouts['actions']

    print(f'From {pkl_file}')
    print(f'Loaded observations: {obs.shape}')
    print(f'Loaded actions: {acs.shape}')
    return ExpertDataset(obs, acs, batch_size)

  def reset(self, obs, acs, batch_size):
    self.obs = obs
    self.acs = acs
    self.batch_size = batch_size
    obs_tensor = th.from_numpy(self.obs)
    acs_tensor = th.from_numpy(self.acs)

    dataset = data.TensorDataset(obs_tensor, acs_tensor)
    self.data_loader = data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    self.iter = iter(self.data_loader)


  def merge(self, obs, acs):
    all_obs = np.vstack((self.obs, obs))
    all_acs = np.vstack((self.acs, acs))
    self.reset(all_obs, all_acs, self.batch_size)

  @property
  def obs_dim(self):
    return self.obs.shape[-1]

  @property
  def acs_dim(self):
    return self.acs.shape[-1]

  def __len__(self):
    return self.obs.shape[0]

  def __iter__(self):
    self.iter = iter(self.data_loader)
    return self

  def __next__(self):
    try:
      return next(self.iter)
    except StopIteration:
      self.iter = iter(self.data_loader)
      return next(self.iter)


class ExpertPolicy:
  """Wraps pre-trained TF expert agent from hw2/experts."""
  def __init__(self, file_path):
    self.policy_fn = load_policy.load_policy(file_path)

  def get_action(self, obs_np):
    with tf.Session():
      tf_util.initialize()
      return self.policy_fn(obs_np)


TrainParams = collections.namedtuple(
  'TrainParams',
  'num_steps mini_batch_size '
  'steps_per_policy_eval policy_eval_eps '
  'lr '
  'steps_between_relabels relabel_batch_size '
)
TrainParams.__new__.__defaults__ = (None,) * len(TrainParams._fields)


class Experiment(experiment.Experiment):
  def __init__(self, env_name, train_params, model_dict):
    env = environment.Environment(env_name)
    super().__init__(env, model.ContinuousActionModel, model_dict,
                     policy.ImitationPolicy, {'lr': train_params.lr})

    self.tp = train_params
    if self.tp.steps_between_relabels:
      assert self.tp.relabel_batch_size

    # Load expert dataset and policy
    pkl_file = f'expert_rollouts/{env_name}/n20_1.pkl'
    self.expert_ds = ExpertDataset.from_pkl(
      pkl_file, batch_size=train_params.mini_batch_size)
    expert_file = os.path.join('experts', env_name + '.pkl')
    self.expert_policy = ExpertPolicy(expert_file)


  def train(self):
    for i in range(self.tp.num_steps):

      # DAgger: rollout current policy, query expert for correct actions,
      # and add them to the dataset.
      relabel_interval = self.tp.steps_between_relabels
      if relabel_interval and i > 0 and i % relabel_interval == 0:
        eps_batch = self.env.sample_rollouts(
          self.policy.get_action, batch_size=self.tp.relabel_batch_size)
        obs = np.array([sar.s for eps in eps_batch for sar in eps])
        expert_acs = self.expert_policy.get_action(obs)
        self.expert_ds.merge(obs, expert_acs)
        self.plt.add_data('ds_size', i, len(self.expert_ds))

      obs_batch, acs_batch = next(self.expert_ds)
      metrics = self.policy.update(obs_batch, acs_batch)
      for name, values in metrics.items():
        self.plt.add_data(name, i, values)

      if i % self.tp.steps_per_policy_eval == 0:
        self.snapshots.add(i, self.policy)
        eps = self.env.sample_rollouts(self.policy.get_action,
                                       num_episodes=self.tp.policy_eval_eps)
        r_per_eps = [sum([sar.r for sar in ep]) for ep in eps]
        self.plt.add_data('r_per_eps', i, r_per_eps)
        print(f'Step: {i}; Loss: {metrics["loss"]}; R: {np.mean(r_per_eps)}')

    self.plt.line_plot()
    self.plt.render()

exps = {}
