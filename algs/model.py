import collections
import functools
import math

import torch as th

from rl.algs import util


def _construct_nn(dims, fn=th.nn.ReLU, last_fn=None):
  layers = []
  for i in range(1, len(dims)):
    layers.append(th.nn.Linear(dims[i - 1], dims[i]))
    activation_fn = fn if i < len(dims) - 1 else last_fn
    if activation_fn:
      layers.append(activation_fn())
  return th.nn.Sequential(*layers)


class DiscreteActionModel(th.nn.Module):
  def __init__(self, obs_dim, acs_dim, hidden_layers=(64,)):
    super().__init__()

    self.nn = _construct_nn(
      (obs_dim,) + hidden_layers + (acs_dim,),
      last_fn=functools.partial(th.nn.LogSoftmax, dim=1))
    self.cuda()

  def forward(self, x):
    return self.nn(x)

  def get_action(self, obs_np):
    """Samples an action to be taken by the policy given observations.

    Args:
      obs_np: numpy array of observations. Size: [obs_dim]

    Returns:
      Integer action to take. Size: [].
    """
    obs_var = util.to_variable(obs_np)
    log_probs = self(obs_var.unsqueeze(0)).squeeze()
    probs = th.exp(log_probs)
    dist = th.distributions.Categorical(probs)
    ac = dist.sample()
    out_np = util.to_numpy(ac).squeeze()
    return out_np

  def log_probs(self, obs_var, acs_var, metrics=None):
    """Computes the log prob that when given obs, model takes the given actions.

    Args:
      obs_var: Variable with observations. Size: [batch, obs_dim]
      acs_var: Variable with actions. Size: [batch]
    Returns:
      log prob of taking the acs under the model, given obs. Size: [batch].
    """
    acs_var = acs_var.unsqueeze(dim=1)  # [batch]  -> [batch, 1].
    log_probs = self(obs_var)
    log_probs = th.gather(log_probs, dim=1, index=acs_var)
    log_probs = log_probs.squeeze()  # [batch, 1] -> [batch].
    return log_probs


class ContinuousActionModel(th.nn.Module):
  def __init__(self, obs_dim, acs_dim,
               shared_layers=(64, 64, 64),
               action_layers=(64,),
               model_std=False,
               min_std=None):
    super().__init__()

    base_nn_dims = (obs_dim,) + shared_layers
    self.base_nn = _construct_nn(base_nn_dims, last_fn=th.nn.ReLU)
    mean_std_dims = (shared_layers[-1],) + action_layers + (acs_dim,)
    self.means = _construct_nn(mean_std_dims)
    if model_std:
      self.logstds = _construct_nn(mean_std_dims)
    else:
      # Must assign th.nn.Parameter to self to be included in parameters().
      self._logstds = th.nn.Parameter(th.zeros(1, acs_dim).cuda())
      self.logstds = lambda _: self._logstds
    self.min_logstd = math.log(min_std) if min_std else None
    self.cuda()

  def forward(self, x):
    x = self.base_nn(x)
    means, logstds = self.means(x), self.logstds(x)
    if self.min_logstd is not None:
      logstds = th.clamp(logstds, min=self.min_logstd)
    return means, logstds

  def get_action(self, obs_np):
    """Samples an action to be taken by the policy given observations.

    Args:
      obs_np: numpy array of observations. Size: [obs_dim].

    Returns:
      Action to take. Size: [acs_dim].
    """
    obs_var = util.to_variable(obs_np)
    dist = self._get_action_distribution(obs_var)
    ac = dist.sample()
    out_np = util.to_numpy(ac)
    return out_np

  def _get_action_distribution(self, obs_var):
    means, logstds = self(obs_var)
    stds = th.exp(logstds)
    dist = th.distributions.Normal(means, stds)
    return dist

  def log_probs(self, obs_var, acs_var, metrics=None):
    """Computes the log prob that when given obs, model takes the given actions.

    Args:
      obs_var: Variable with observations. Size: [batch, obs_dim]
      acs_var: Variable with actions. Size: [batch, acs_dim]
    Returns:
      log prob of taking the acs under the model, given obs. Size: [batch].
    """
    dist = self._get_action_distribution(obs_var)
    log_probs = dist.log_prob(acs_var)
    log_probs = log_probs.squeeze()  # [batch, 1] -> [batch].
    if metrics is not None:
      metrics['ac_mean'] = util.to_numpy(dist.mean)
      metrics['ac_std'] = util.to_numpy(dist.std)
    return log_probs


class ValueNetwork(th.nn.Module):
  def __init__(self, obs_dim, hidden_layers=(64,)):
    super().__init__()

    self.nn = _construct_nn((obs_dim,) + hidden_layers + (1,))
    self._init_weights()
    self.cuda()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, th.nn.Linear):
        th.nn.init.xavier_normal(m.weight, gain=1e-2)
        m.bias.data.zero_()

  def forward(self, x):
    return self.nn(x)


ConvSpec = collections.namedtuple('ConvSpec', ['w', 's'])

class QNetwork(th.nn.Module):
  """Q-network with convs and fc, with one action-value output per action.

  """
  def __init__(self, obs_dim, acs_dim):
    """Creates a Q-network.

    Args:
      obs_dim: shape of the observation tensor. Assumed to be HWC.
      acs_dim: number of discrete actions.
    """
    super().__init__()

    self.convs = th.nn.Sequential(
      th.nn.Conv2d(obs_dim[-1], 32, 8, 4),
      th.nn.ReLU(),
      th.nn.Conv2d(32, 64, 4, 2),
      th.nn.ReLU(),
      th.nn.Conv2d(64, 64, 3, 1),
      th.nn.ReLU(),
    )
    self.fc = _construct_nn((64*7*7, 512, acs_dim))
    self.cuda()

  def forward(self, x):
    """Runs the model on input observations.

    Args:
      x: 4-D image tensor. Assumed to be BHWC.

    Returns:
      Softmax probabilities, shaped [B, acs_dim]
    """
    x = x.permute(0, 3, 1, 2)  # Convert to BCHW.
    x = self.convs(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
