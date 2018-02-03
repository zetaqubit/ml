"""Policy gradient experiments for discrete and continuous action spaces."""

import collections

from rl.algs import environment
from rl.algs import experiment
from rl.algs import model
from rl.algs import policy

TrainParams = collections.namedtuple(
  'TrainParams',
  'num_train_iters '
  'steps_per_batch '
  'steps_per_snapshot '
  'lr '
)


class Experiment(experiment.Experiment):
  def __init__(self, env_name, train_params, model_params):
    self.tp = train_params
    is_discrete = environment.Environment(env_name).discrete_ac
    model_cls = (model.DiscreteActionModel if is_discrete
    else model.ContinuousActionModel)
    super().__init__(env_name, model_cls, model_params, policy.PolicyGradient,
                     {'lr': self.tp.lr})

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
