from pprint import pprint

import gym
import numpy as np

# (0, 0) is top-left. Array is row-major, so (r, c).
_ACTIONS = [
  (-1, 0, 0),  # up
  (0, 1, 0),  # right
  (1, 0, 0),  # down
  (0, -1, 0),  # left
]

_NUM_ACTIONS = len(_ACTIONS)

_EMPTY = 0
_CURRENT = 1
_END = 2
_LAST = _END  # Sentinel.


class GridWorld(gym.core.Env):
  def __init__(self, grid_shape=(4, 4, 1)):
    self._grid_shape = grid_shape
    self._min_pos = np.zeros_like(grid_shape)
    self._max_pos = np.array(grid_shape) - 1
    self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
    self.observation_space = gym.spaces.Box(low=_EMPTY, high=_LAST,
                                            shape=grid_shape, dtype=np.uint8)
    self.seed()

  def seed(self, seed=None):
    self._random_state = np.random.RandomState(seed)
    self.reset()

  def reset(self):
    self._state = np.full(self._grid_shape, _EMPTY)

    self.current_pos, self.end_pos = self._sample_locations(n=2)
    self._state[self.current_pos] = _CURRENT
    self._state[self.end_pos] = _END
    return self._state.copy()

  def _sample_locations(self, n=1):
    """Samples n locations, without replacement."""
    choices = np.prod(self._grid_shape)
    flat_indices = self._random_state.choice(choices, n, replace=False)
    indices = [tuple(np.unravel_index(flat_index, self._grid_shape))
               for flat_index in flat_indices]
    return indices

  def step(self, action):
    assert 0 <= action < _NUM_ACTIONS
    if self._is_done():
      print('Already in terminating state.')
      return self._state.copy(), 0, True, None

    self._state[self.current_pos] = _EMPTY
    new_pos = np.array(self.current_pos) + _ACTIONS[action]
    self.current_pos = self._clip_pos(new_pos)
    self._state[self.current_pos] = _CURRENT
    r = 0 if self._is_done() else -1
    return self._state.copy(), r, self._is_done(), None

  def _is_done(self):
    return self.current_pos == self.end_pos

  def _clip_pos(self, pos):
    return tuple(np.clip(pos, self._min_pos, self._max_pos))

  def render(self, mode='human'):
    pprint(self._state.squeeze())


