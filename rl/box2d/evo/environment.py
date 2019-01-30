"""2D Physics environment, based on Box2D.

Provides per-frame simulation, event callbacks, and input handling.
"""
import numpy as np
import pygame

from rl.box2d.app import framework

from Box2D import b2

class Agent:
  def __init__(self, body):
    self._body = body

  def move(self, forces):
    """Applies forces to move the agent.

    Args:
     forces: array of forces to apply.
       [0]: forward/backward
       [1]: left/right torque
    """
    # cm = self._body.centerOfMass
    b = self._body
    forward = b.GetWorldVector((0, 1))
    b.ApplyForceToCenter(forces[0] * forward, wake=True)
    b.ApplyTorque(forces[1], wake=True)

class Environment(framework.Framework):
  name = "2D Environment"  # Name of the class to display
  description = "2D physics simulation"

  # Play area range is [-20, 20] x [-20, 20].
  _PLAY_AREA_SIZE = np.array([40, 40])

  _AGENT_MOVE_FORCE = 100
  _AGENT_TURN_FORCE = 30

  # Categories (for collision filtering).
  _CAT_AGENT = 0x1
  _CAT_FOOD = 0x2

  def __init__(self):
    """
    Initialize all of your objects here.
    Be sure to call the Framework's initializer first.
    """
    super().__init__(gravity=(0, 0), screen_size=(800, 1000))

    # wall_body = self.world.CreateBody(position=(0, -10))
    # wall_box = b2.polygonShape(box=(50, 10))
    # wall_body.CreateFixture(shape=wall_box)

    # wall_body = self.world.CreateBody(
    #     position=(0, -10),
    #     shapes=[b2.polygonShape(box=(50, 10)),
    #             b2.polygonShape(box=(10, 50))],
    # )

    # box1 = self.world.CreateDynamicBody(position=(0, 0))
    # box1.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)
    #
    # box2 = self.world.CreateDynamicBody(
    #     position=(3, 4),
    #     angle=0,
    #     linearDamping=1,
    #     angularDamping=1,
    #     shapes=b2.polygonShape(box=(1, 1)),
    #     shapeFixture=b2.fixtureDef(density=1, friction=0.3)
    # )

    # The boundaries.
    dx, dy = self._PLAY_AREA_SIZE / 2.0
    play_area = self.world.CreateBody(position=(0, 0))
    play_area.CreateEdgeChain(
        [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy), (-dx, -dy)]
    )

    self._main_agent = self._spawn_agent((0, 0))
    self._spawn_agent((5, 0))

    self._to_destroy = set()

  def _unit_to_world(self, unit_coords):
    return self._PLAY_AREA_SIZE * (unit_coords - 0.5)

  def _spawn_food(self):
    pos = self._unit_to_world(np.random.rand(2))
    food = self.world.CreateBody(position=pos)
    food.CreateCircleFixture(radius=0.4,
                             categoryBits=self._CAT_FOOD)

  def _spawn_agent(self, pos=(0, 0)):
    body = self.world.CreateDynamicBody(
        position=pos, angle=0, linearDamping=0.5, angularDamping=1)
    body.CreatePolygonFixture(vertices=[(-1, 0), (0, 3), (1, 0)], density=1, friction=0.3,
                              categoryBits=self._CAT_AGENT)
    return Agent(body)

  def Keyboard(self, key):
    """
    The key is from Keys.K_*
    (e.g., if key == Keys.K_z: ... )
    """
    if key == pygame.K_s:
      self._spawn_food()


  def CheckKeys(self):
    super().CheckKeys()

    # ESDF in Dvorak for movement.
    move_force = np.zeros(2)
    if self.keys[pygame.K_PERIOD]:
      move_force[0] += self._AGENT_MOVE_FORCE
    if self.keys[pygame.K_e]:
      move_force[0] -= self._AGENT_MOVE_FORCE
    if self.keys[pygame.K_o]:
      move_force[1] += self._AGENT_TURN_FORCE
    if self.keys[pygame.K_u]:
      move_force[1] -= self._AGENT_TURN_FORCE
    self._main_agent.move(move_force)


  def Step(self, settings):
    """Called upon every step.
    You should always call
     -> super().Step(settings)
    at the beginning or end of your function.

    If placed at the beginning, it will cause the actual physics step to happen first.
    If placed at the end, it will cause the physics step to happen after your code.
    """

    super().Step(settings)

    # Destroy bodies pending destruction.
    for body in self._to_destroy:
      self.world.DestroyBody(body)
    self._to_destroy.clear()

  def BeginContact(self, contact):
    print(contact)
    if self._is_type(contact.fixtureA, self._CAT_FOOD):
      self._to_destroy.add(contact.fixtureA.body)
    if self._is_type(contact.fixtureB, self._CAT_FOOD):
      self._to_destroy.add(contact.fixtureB.body)

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


if __name__ == "__main__":
  framework.main(Environment)
