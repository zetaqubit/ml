"""Tests for rl.box2d.newton.box2d_state."""

from Box2D import b2
import pytest

from rl.box2d.newton import box2d_state


@pytest.fixture
def state1():
  return box2d_state.ObjectState(position=(1, 2))

@pytest.fixture
def state2():
  return box2d_state.ObjectState(position=(1, 2))

@pytest.fixture
def world_state(state1, state2):
  return box2d_state.WorldState(objects={
      'a': state1,
      'b': state2,
  }, gravity=(0, -9.81))

def test_object_state_construct(state1):
  world = b2.world()
  obj = state1.construct(world)
  assert obj.position == state1.position
  assert obj.linearVelocity == state1.velocity
  assert obj.angle == state1.angle
  assert obj.angularVelocity == state1.angular_velocity
  assert obj.linearDamping == state1.linear_damping
  assert obj.angularDamping == state1.angular_damping
  assert obj.fixedRotation == state1.fixed_rotation

  assert obj.fixtures[0].friction == state1.friction
  assert obj.fixtures[0].density == state1.density
  assert obj.fixtures[0].restitution == state1.restitution

def test_world_state_construct(world_state, state1, state2):
  world = world_state.construct()
  assert world.gravity == world_state.gravity
  assert world.bodyCount == 2
  expected_world = b2.world()
  expected_a = state1.construct(expected_world)
  assert world.objects['a'].position == expected_a.position
  expected_b = state2.construct(expected_world)
  assert world.objects['b'].position == expected_b.position

def test_object_state_json(state1):
  json = state1.to_json()
  deserialized = box2d_state.ObjectState.from_json(json)
  assert deserialized == state1

def test_world_state_json(world_state):
  json = world_state.to_json()
  deserialized = box2d_state.WorldState.from_json(json)
  assert deserialized == world_state

