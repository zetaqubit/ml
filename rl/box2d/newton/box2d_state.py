"""Box2D state serialization/deserialization.
"""

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Dict, Tuple

from Box2D import b2
import numpy as np


@dataclass
@dataclass_json
class ObjectState:
  # Body
  position: Tuple[float] = (0, 0)
  velocity: Tuple[float] = (0, 0)
  angle: float = 0
  angular_velocity: float = 0
  linear_damping: float = 0
  angular_damping: float = 0
  fixed_rotation: bool = False

  # Shape
  vertices: Tuple[Tuple[float]] = ((-1, 1), (1, 1), (1, -1), (-1, -1))
  vertices_scale: float = 1

  # Fixture
  density: float = 1
  friction: float = 0
  restitution: float = 1  # [0, 1] <-> [inelastic, elastic]

  def construct(self, world):
    """Creates Box2D object with this state."""
    b2_obj = world.CreateDynamicBody()
    b2_obj.position = self.position
    b2_obj.linearVelocity = self.velocity
    b2_obj.angle = self.angle
    b2_obj.angularVelocity = self.angular_velocity
    b2_obj.linearDamping = self.linear_damping
    b2_obj.angularDamping = self.angular_damping
    b2_obj.fixedRotation = self.fixed_rotation

    vertices = (self.vertices_scale * np.array(self.vertices)).tolist()
    b2_obj.CreatePolygonFixture(
        vertices=vertices,
        density=self.density,
        restitution=self.restitution,
        friction=self.friction,
    )
    return b2_obj

@dataclass
@dataclass_json
class WorldState:
  objects: Dict[str, ObjectState]
  gravity: Tuple[float] = (0, 0)

  def construct(self, world=None):
    """Creates a b2.world from the WorldState.

    Populates world.objects with specified objects.

    Example access:
      world.objects['a']: the b2.body for self.objects['a']
      world.gravity: b2.vec2 of gravity direction
    """
    if world is None:
      world = b2.world()
    world.gravity = self.gravity
    objects = {
      name: state.construct(world)
      for name, state in self.objects.items()
    }
    setattr(world, 'objects', objects)
    return world


def dynamic_object_state(b2_obj):
  state = {}
  pos = b2_obj.position
  state['pos_x'] = pos[0]
  state['pos_y'] = pos[1]
  state['pos_a'] = b2_obj.angle

  vel = b2_obj.linearVelocity
  state['vel_x'] = vel[0]
  state['vel_y'] = vel[1]
  state['vel_a'] = b2_obj.angularVelocity
  return state


def dynamic_world_state(world):
  state = {}
  for name, b2_obj in world.objects.items():
    obj_state = dynamic_object_state(b2_obj)
    obj_state = {name + '.' + k: v for k, v in obj_state.items()}
    state.update(obj_state)
  return state
