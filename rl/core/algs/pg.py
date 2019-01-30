"""Policy gradient experiments for discrete and continuous action spaces."""

import collections

import numpy as np

from rl.core.envs import environment
from rl.core.algs import util, experiment, policy, model

TrainParams = collections.namedtuple(
  'TrainParams',
  'num_train_iters '
  'steps_per_batch '
  'steps_per_snapshot '
  'lr '
  'discount '
)


class Experiment(experiment.Experiment):
  def __init__(self, env_name, train_params, model_params,
               value_nn_params=None):
    self.tp = train_params
    env = environment.Environment(env_name)
    model_cls = (model.DiscreteActionModel if env.discrete_ac
                 else model.ContinuousActionModel)
    policy_kwargs = {'lr': self.tp.lr, 'discount': self.tp.discount}
    if value_nn_params is not None:
      value_nn_kwargs = dict(value_nn_params)
      value_nn_kwargs.update({
        'obs_dim': env.obs_dim,
      })
      value_nn = model.ValueNetwork(**value_nn_kwargs)
      policy_kwargs['value_nn'] = value_nn
    super().__init__(env, model_cls, model_params, policy.PolicyGradient,
                     policy_kwargs)

    self.plt.writer.add_graph(self.policy.model,
                              util.to_tensor(np.zeros((1, env.obs_dim))))
    if value_nn_params is not None:
      self.plt.writer.add_graph(value_nn,
                                util.to_tensor(np.zeros((1, env.obs_dim))))

  def train(self):
    for i in range(self.tp.num_train_iters):
      eps_batch = self.env.sample_rollouts(self.policy.get_action,
                                           self.tp.steps_per_batch)
      metrics = self.policy.update(eps_batch)

      if i % self.tp.steps_per_snapshot == 0:
        for name, m in metrics.items():
          self.plt.add_data(name, i, m)
        self.snapshots.add(i, self.policy)

    self.plt.line_plot()
    self.plt.render()


exps = {}
