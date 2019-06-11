"""Simulation where objects collide, demonstrating conservation of momentum.

Research goal:
- Simulation provides video sequences of colliding objects
- Model learns to extract latent physical quantities (mass, velocity)
  and use them to predict future states.

Observation:
- frame_0:frame_k: Initial k video frames of object trajectories

Prediction:
- frame_t: Video frame at time t.

(Unobserved) Parameters:
- m1, v1_0: mass, initial velocity of object 1
- m2, v2_0: mass, initial velocity of object 2

(Latent) Predictions:
- v1_t: velocity of object 1 at time t
- v2_t: velocity of object 2 at time t
"""

import numpy as np

from rl.box2d.app import framework
from rl.box2d.newton import box2d_log

from Box2D import b2


class MomentumSim(framework.Framework):
  name = 'Momentum Simulation'
  description = 'Demonstrates conservation of momentum'

  # Play area range is [-20, 20] x [-20, 20].
  _PLAY_AREA_SIZE = np.array([40, 40])

  def __init__(self):
    super().__init__(gravity=(0, 0), screen_size=(800, 1000))

    # Initial conditions.
    self._obj1 = self._create_object((-10, 0))
    self._obj2 = self._create_object(pos=(10, 0), size=(2, 2))
    self._apply_impulse(self._obj1, [120, 0])

    self._log = box2d_log.CsvLog(log_dir='~/code/data/newton/momentum/')
    self.log_state()

  def _unit_to_world(self, unit_coords):
    return self._PLAY_AREA_SIZE * (unit_coords - 0.5)

  def _create_object(self, pos=(0, 0), size=(2, 2)):
    body = self.world.CreateDynamicBody(
        position=pos, angle=0, linearDamping=0.5, angularDamping=1)
    w, h = size[0] / 2, size[1] / 2
    body.CreatePolygonFixture(
        vertices=[(-w, -h), (-w, h), (w, h), (w, -h)], density=1, friction=0.3,
        restitution=1)
    return body

  def _apply_impulse(self, body, impulse):
    body.ApplyLinearImpulse(b2.vec2(impulse), body.worldCenter, wake=True)


  def Step(self, settings):
    """Called upon every step.
    You should always call
     -> super().Step(settings)
    at the beginning or end of your function.

    If placed at the beginning, it will cause the actual physics step to happen first.
    If placed at the end, it will cause the physics step to happen after your code.
    """
    super().Step(settings)

    self.log_state()

  def log_state(self):
    state = {'step': self.stepCount}
    state.update(box2d_log.object_state(self._obj1, '1'))
    state.update(box2d_log.object_state(self._obj2, '2'))
    self._log.add(state)

  def BeginContact(self, contact):
    print(contact)

  def ShapeDestroyed(self, shape):
    """
    Callback indicating 'shape' has been destroyed.
    """
    pass

  def _is_type(self, fixture, category_mask):
    return (fixture.filterData.categoryBits & category_mask) > 0

  def JointDestroyed(self, joint):
    """
    The joint passed in was removed.
    """
    pass

  def OnExit(self):
    self._log.write()
    print(f'Wrote {self._log.num_events()} events to {self._log.filepath}')


if __name__ == "__main__":
  framework.main(MomentumSim)
