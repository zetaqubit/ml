"""Tests for rl.box2d.newton.box2d_state."""

import unittest

from Box2D import b2

from rl.box2d.newton import box2d_state


class Box2dStateTest(unittest.TestCase):
  def setUp(self):
    self.state1 = box2d_state.ObjectState(position=(1, 2))
    self.state2 = box2d_state.ObjectState(position=(3, 4))
    self.world_state = box2d_state.WorldState(objects={
        'a': self.state1,
        'b': self.state2,
    }, gravity=(0, -9.81))

  def test_object_state_construct(self):
    world = b2.world()
    obj = self.state1.construct(world)
    self.assertEqual(self.state1.position, obj.position)
    self.assertEqual(self.state1.velocity, obj.linearVelocity)
    self.assertEqual(self.state1.angle, obj.angle)
    self.assertEqual(self.state1.angular_velocity, obj.angularVelocity)
    self.assertEqual(self.state1.linear_damping, obj.linearDamping)
    self.assertEqual(self.state1.angular_damping, obj.angularDamping)
    self.assertEqual(self.state1.fixed_rotation, obj.fixedRotation)

    self.assertEqual(self.state1.friction, obj.fixtures[0].friction)
    self.assertEqual(self.state1.density, obj.fixtures[0].density)
    self.assertEqual(self.state1.restitution, obj.fixtures[0].restitution)

  def test_world_state_construct(self):
    world = self.world_state.construct()
    self.assertEqual(self.world_state.gravity, world.gravity)
    self.assertEqual(2, world.bodyCount)
    expected_world = b2.world()
    expected_a = self.state1.construct(expected_world)
    self.assertEqual(expected_a.position, world.objects['a'].position)
    expected_b = self.state2.construct(expected_world)
    self.assertEqual(expected_b.position, world.objects['b'].position)

  def test_object_state_json(self):
    json = self.state1.to_json()
    deserialized = box2d_state.ObjectState.from_json(json)
    self.assertEqual(self.state1, deserialized)

  def test_world_state_json(self):
    json = self.world_state.to_json()
    deserialized = box2d_state.WorldState.from_json(json)
    self.assertEqual(self.world_state, deserialized)


if __name__ == '__main__':
  unittest.main()