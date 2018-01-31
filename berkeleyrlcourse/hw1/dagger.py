# Section 2 and 3: Behavior Cloning.

import collections
import pickle

import numpy as np
import torch as th
from torch import autograd as tha
from torch.utils import data

from rl.algs import experiment
from rl.algs import pg

dtype = th.cuda.FloatTensor


class ExpertDataset(object):
  def __init__(self, pkl_file, batch_size):
    with open(pkl_file, 'rb') as fd:
      rollouts = pickle.load(fd)
    self.obs = rollouts['observations']
    self.acs = rollouts['actions']

    print(f'From {pkl_file}')
    print(f'Loaded observations: {self.obs.shape}')
    print(f'Loaded actions: {self.acs.shape}')

    obs_tensor = th.from_numpy(self.obs).cuda()
    acs_tensor = th.from_numpy(self.acs).cuda()

    dataset = data.TensorDataset(obs_tensor, acs_tensor)
    self.data_loader = data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True)

  @property
  def obs_dim(self):
    return self.obs.shape[-1]

  @property
  def acs_dim(self):
    return self.acs.shape[-1]

  def __iter__(self):
    self.iter = iter(self.data_loader)
    return self

  def __next__(self):
    try:
      return next(self.iter)
    except StopIteration:
      self.iter = iter(self.data_loader)
      return next(self.iter)


class ImitationPolicy(object):
  """Continuous action policy, trained on imitation."""
  def __init__(self, model, lr=0.001):
    self.model = model
    self.optimizer = th.optim.Adam(self.model.parameters(), lr)

  def get_action(self, obs_np):
    return self.model.get_action(obs_np)

  def step(self, obs_batch, acs_batch):
    metrics = {}
    obs_var = tha.Variable(obs_batch.type(dtype))
    acs_var = tha.Variable(acs_batch.type(dtype))
    log_probs = self.model.log_probs(obs_var, acs_var, metrics)
    loss = -log_probs.mean()

    metrics['loss'] = loss.data.cpu().numpy()[0]

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return metrics

TrainParams = collections.namedtuple(
  'TrainParams',
  'num_steps mini_batch_size steps_per_policy_eval '
  'policy_eval_eps lr'
)


class Experiment(experiment.Experiment):
  def __init__(self, env_name, train_params, model_dict):
    super().__init__(env_name, pg.ContinuousActionModel, model_dict,
                     ImitationPolicy, {'lr': train_params.lr})

    self.tp = train_params
    pkl_file = f'expert_rollouts/{env_name}/n100_1.pkl'
    self.expert_ds = ExpertDataset(
      pkl_file, batch_size=train_params.mini_batch_size)

  def train(self):
    for i, (obs_batch, acs_batch) in zip(range(self.tp.num_steps), self.expert_ds):
      metrics = self.policy.step(obs_batch, acs_batch)
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
