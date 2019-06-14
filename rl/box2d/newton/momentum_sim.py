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

from rl.box2d.app import framework
from rl.box2d.newton import box2d_log
from rl.box2d.newton import box2d_state


WORLD_STATE_JSON = """
{
  'gravity': [0, 0],
  'objects': {
    'obj1': {
      'position': [-10, 0],
      'velocity': [10, 0]
    },
    'obj2': {
      'position': [10, 0],
      'vertices_scale': 2
    }
  }
}
""".replace("'", '"')


class MomentumSim(framework.Framework):
  name = 'Momentum Simulation'
  description = 'Demonstrates conservation of momentum'

  def __init__(self, world_state_json=WORLD_STATE_JSON):
    super().__init__(screen_size=(800, 1000))

    world_state = box2d_state.WorldState.from_json(world_state_json)
    world_state.construct(self.world)

    self._log = box2d_log.Box2DLog(
        self.world, log_dir='~/code/data/newton/momentum/')
    self._log.add_world_state(self.stepCount)

  def Step(self, settings):
    """Called upon every step.
    You should always call
     -> super().Step(settings)
    at the beginning or end of your function.

    If placed at the beginning, it will cause the actual physics step to happen first.
    If placed at the end, it will cause the physics step to happen after your code.
    """
    super().Step(settings)
    self._log.add_world_state(self.stepCount)

  def BeginContact(self, contact):
    pass

  def OnExit(self):
    self._log.write()
    print(f'Wrote {self._log.num_events()} events to {self._log.filepath}')


if __name__ == "__main__":
  framework.main(MomentumSim)
