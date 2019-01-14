import collections
from dataclasses import dataclass
import functools
import math

import numpy as np
import torch as th

from rl.core.algs import util


@dataclass(frozen=True)
class ConvSpec:
  depth: int
  width: int
  stride: int = 1
  padding: int = 0


# Network used in the Atari DQN paper.
DQN_CONV_SPECS = [
  ConvSpec(32, 8, 4),
  ConvSpec(64, 4, 2),
  ConvSpec(64, 3, 1),
]

DQN_FC_SPECS = (512,)


def _fc_nn(dims, fn=th.nn.ReLU, last_fn=None):
  layers = []
  for i in range(1, len(dims)):
    layers.append(th.nn.Linear(dims[i - 1], dims[i]))
    activation_fn = fn if i < len(dims) - 1 else last_fn
    if activation_fn:
      layers.append(activation_fn())
  return th.nn.Sequential(*layers)


def cnn(conv_specs, in_depth, fn=th.nn.ReLU, last_fn=None, bn=False):
  """Creates a CNN model with the given specs.

  Args:
    conv_specs: list of `ConvSpec`s defining the CNN.
    in_depth: input channels.
    fn: activation function of every layer, except last layer.
    last_fn: activation function of the last layer.
    bn: whether to use batch normalization after each activation, except
      last layer.
  Returns:
    A torch.Model implementing the CNN
  """
  layers = []
  last_depth = in_depth
  for i, spec in enumerate(conv_specs):
    layers.append(th.nn.Conv2d(last_depth, spec.depth, spec.width,
                               spec.stride, spec.padding))
    not_last_layer = i < len(conv_specs) - 1

    activation_fn = fn if not_last_layer else last_fn
    if activation_fn:
      layers.append(activation_fn())

    if bn and not_last_layer:
      layers.append(th.nn.BatchNorm2d(spec.depth))

    last_depth = spec.depth
  return th.nn.Sequential(*layers)


def cnn_shape(conv_specs, in_shape):
  """Calculates the output shape after running CNN on image of in_shape.

  Args:
    conv_specs: list of `ConvSpec`s defining the CNN.
    in_shape: shape of the input image, as [H, W, C].
  Returns:
    The output shape as a tuple, as [H, W, C].
  """
  out_shape = list(in_shape)

  def calc_width(in_w, kernel_w, stride, padding):
    return math.floor((in_w + 2 * padding - kernel_w) / stride + 1)

  for spec in conv_specs:
    out_shape[-1] = spec.depth
    out_shape[-2] = calc_width(out_shape[-2], spec.width, spec.stride,
                               spec.padding)
    out_shape[-3] = calc_width(out_shape[-3], spec.width, spec.stride,
                               spec.padding)
  return tuple(out_shape)



class DiscreteActionModel(th.nn.Module):
  def __init__(self, obs_dim, acs_dim, hidden_layers=(64,)):
    super().__init__()

    self.nn = _fc_nn(
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
    obs_var = util.to_tensor(obs_np)
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
    self.base_nn = _fc_nn(base_nn_dims, last_fn=th.nn.ReLU)
    mean_std_dims = (shared_layers[-1],) + action_layers + (acs_dim,)
    self.means = _fc_nn(mean_std_dims)
    if model_std:
      self.logstds = _fc_nn(mean_std_dims)
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
    obs_var = util.to_tensor(obs_np)
    dist = self._get_action_distribution(obs_var.unsqueeze(0))
    ac = dist.sample().squeeze(0)
    out_np = util.to_numpy(ac)
    return out_np

  def _get_action_distribution(self, obs_var):
    """Computes the action distribution given observations.

    Args:
      obs_var: Tensor of observations. Size: [batch, obs_dim].

    Returns:
      Distribution over actions. Sampled size: [batch, acs_dim].
    """
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
      metrics['ac_std'] = util.to_numpy(dist.stddev)
    return log_probs


class ValueNetwork(th.nn.Module):
  def __init__(self, obs_dim, hidden_layers=(64,)):
    super().__init__()

    self.nn = _fc_nn((obs_dim,) + hidden_layers + (1,))
    self._init_weights()
    self.cuda()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, th.nn.Linear):
        th.nn.init.xavier_normal(m.weight, gain=1e-2)
        m.bias.data.zero_()

  def forward(self, x):
    return self.nn(x)


class QNetwork(th.nn.Module):
  """Q-network with convs and fc, with one action-value output per action.

  """
  def __init__(self, obs_dim, acs_dim, conv_specs, fc_specs):
    """Creates a Q-network.

    Args:
      obs_dim: shape of the observation tensor. Assumed to be HWC.
      acs_dim: number of discrete actions.
      conv_specs: list of convolution layer specs.
      fc_specs: list of fully-connected layer specs. These follow the convs.
    """
    super().__init__()

    self.convs = cnn(conv_specs, obs_dim[-1], fn=th.nn.ReLU, last_fn=th.nn.ReLU)
    conv_out_shape = cnn_shape(conv_specs, obs_dim)
    print(conv_out_shape)
    assert (np.array(conv_out_shape) > 0).all()
    fc_in_size = int(np.product(conv_out_shape))
    fc_sizes = (fc_in_size,) + fc_specs + (acs_dim,)
    self.fc = _fc_nn(fc_sizes)
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

  def action_values(self, obs_np):
    obs_var = util.to_tensor(obs_np)
    return util.to_numpy(self(obs_var))

  def get_action(self, obs_np):
    qs = self.action_values(obs_np)
    return np.argmax(qs, -1)

