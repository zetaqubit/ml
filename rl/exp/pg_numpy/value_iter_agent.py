from __future__ import division
from __future__ import print_function

from gym.envs.toy_text import frozen_lake
import numpy as np

# (0, 0) is top left
# Indexed by (row, col)

_ACTIONS = [
  (0, -1),  # left
  (1, 0),   # down
  (0, 1),   # right
  (-1, 0),  # up
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
    if state_size[0] == 4:
      self._map = frozen_lake.MAPS['4x4']

  def env_transition_prob_estimated(self, loc, action):
    counts = self._env_tx[loc + (action,)]
    sum = np.sum(np.sum(counts, axis=-1), axis=-1)
    if sum == 0:
      return np.full(self._action_size, 1 / self._action_size)
      #return counts[new_loc]
    return counts / sum

  def env_transition_prob(self, loc, action):
    transitions = np.zeros_like(self._value)
    map_sym = self._map[loc[0]][loc[1]]
    if map_sym == 'H':
      return transitions
    if map_sym == 'G':
      transitions[0, 0] = 1
      return transitions
    for noisy_a in [(action-1) % 4, action, (action+1) % 4]:
      new_loc = self._move_loc(loc, noisy_a)
      transitions[new_loc] += 1 / 3
    return transitions

  def env_reward(self, loc):
    map_sym = self._map[loc[0]][loc[1]]
    if map_sym == 'G':
      return 1
    elif map_sym == 'H':
      return -1
    return 0

  def observation_to_state(self, obs):
    return (obs // self._state_size[0],
            obs % self._state_size[0])

  def _move_loc(self, loc, action):
    new_loc = np.array(loc) + np.array(_ACTIONS[action])
    new_loc = np.clip(new_loc, 0, self._state_size[0] - 1)
    return tuple(new_loc)

  def _evaluate_policy(self):
    for loc, val in np.ndenumerate(self._value):
      reward = self.env_reward(loc)
      action_probs = self._policy[loc]
      v = 0
      for action, action_prob in enumerate(action_probs):
        env_probs = self.env_transition_prob(loc, action)
        for new_loc, env_prob in np.ndenumerate(env_probs):
          if env_prob == 0:
            continue
          discounted_r = action_prob * env_prob * (
              reward + self._gamma * self._value[new_loc])
          v += discounted_r
      self._value[loc] = v

  def _update_policy(self):
    for loc, val in np.ndenumerate(self._value):
      v_under_as = []
      for action in range(self._action_size):
        v_under_a = 0
        env_probs = self.env_transition_prob(loc, action)
        for new_loc, env_prob in np.ndenumerate(env_probs):
          if env_prob == 0:
            continue
          discounted_r = env_prob * self._gamma * self._value[new_loc]
          v_under_a += discounted_r
        v_under_as.append(v_under_a)

      max_v = np.max(v_under_as)
      max_idx = v_under_as == max_v
      num_max = np.sum(max_idx)
      self._policy[loc] = np.full((self._action_size), self._epsilon)
      remaining_probs = 1 - self._epsilon * (self._action_size - num_max)
      self._policy[loc][max_idx] = remaining_probs / num_max


  def act(self, observation, reward, done):
    self._evaluate_policy()
    self._update_policy()

    location = self.observation_to_state(observation)
    chosen = np.random.choice(range(self._action_size),
                              p=self._policy[location])
    return chosen

  def act2(self, observation, reward, done):
    location = self.observation_to_state(observation)
    action_probs = self._policy[location]
    v = 0
    max_v = None
    max_as = []
    v_under_as = []
    for action, action_prob in enumerate(action_probs):
      v_under_a = 0
      # TODO: factor in environment transition probs
      env_probs = self.env_transition_prob(location, action)
      for new_loc, env_prob in np.ndenumerate(env_probs):
        if env_prob == 0:
          continue
        all_rewards = reward + self._gamma * self._value[new_loc]
        expected_reward = env_prob * all_rewards
        v_under_a += expected_reward

      v_under_as.append(v_under_a)
      v += action_prob * v_under_a
      #v += v_under_a
      if max_v is None or v_under_a > max_v:
        max_v = v_under_a
        max_as = [action]
      elif v_under_a == max_v:
        max_as.append(action)
    self._value[location] = v

    # TODO: update self._policy
    #self._policy[location] = softmax(v_under_as)
    remaining_probs = 1 - self._epsilon * (self._action_size - len(max_as))
    self._policy[location] = np.full((self._action_size), self._epsilon)
    self._policy[location][max_as] = remaining_probs / len(max_as)

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

