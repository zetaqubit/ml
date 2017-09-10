from __future__ import division
from __future__ import print_function

import numpy as np

# (0, 0) is top left
# Indexed by (row, col)

_ACTIONS = [
  (1, 0),   # down
  (0, 1),   # right
  (-1, 0),  # up
  (0, -1),  # left
]

_START_LOC = (0, 0)

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

class ValueIterAgent(object):
  def __init__(self, state_size, action_size):
    self._state_size = state_size
    self._action_size = action_size
    self._value = np.zeros(state_size)
    self._policy = np.full(state_size + (action_size,),
                           1 / action_size)
    self._env_tx = np.ones(state_size + (action_size,) + state_size)
    self._gamma = 0.9
    self._epsilon = 0.1
    self._last_loc = None
    self._last_action = None

  def env_transition_prob(self, loc, action, new_loc):
    counts = self._env_tx[loc + (action,)]
    sum = np.sum(np.sum(counts, axis=-1), axis=-1)
    if sum == 0:
      return np.full(self._action_size, 1 / self._action_size)
      #return counts[new_loc]
    return counts[new_loc] / sum

  def observation_to_state(self, obs):
    return (obs // self._state_size[0],
            obs % self._state_size[0])

  def _move_loc(self, loc, action):
    new_loc = np.array(loc) + np.array(_ACTIONS[action])
    new_loc = np.clip(new_loc, 0, self._state_size[0] - 1)
    return tuple(new_loc)

  def act(self, observation, reward, done):
    location = self.observation_to_state(observation)
    action_probs = self._policy[location]
    v = 0
    max_as = []
    max_v = None
    next_vals = []
    for action, _ in enumerate(_ACTIONS):
      new_loc = self._move_loc(location, action)
      probs = self._policy[(new_loc) + (action,)]
      v_next = self._gamma * self._value[new_loc]
      # TODO: factor in environment transition probs
      env_prob = self.env_transition_prob(location, action, new_loc)
      next_val = probs * env_prob * (reward + v_next)
      next_vals.append(next_val)
      v += next_val
      if max_v is None or v_next > max_v:
        max_v = v_next
        max_as = [action]
      elif v_next == max_v:
        max_as.append(action)
    self._value[location] = v

    # TODO: update self._policy
    self._policy[location] = softmax(next_vals)


    if np.random.random() < self._epsilon:
      chosen = np.random.choice(range(self._action_size))
    else:
      chosen = np.random.choice(range(self._action_size),
                              p=self._policy[location])
      #chosen = np.random.choice(max_as)
    if self._last_loc is not None:
      self._env_tx[self._last_loc + (self._last_action,) + location] += 1
    if done:
      self._last_action = self._last_loc = None
    else:
      self._last_action = chosen
      self._last_loc = location
    return chosen

