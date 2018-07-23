"""2D Physics environment, based on Box2D.

Provides per-frame simulation, event callbacks, and input handling.
"""
from rl.box2d.app import framework


class Environment(framework.Framework):
  name = "2D Environment"  # Name of the class to display
  description = "2D physics simulation"

  def __init__(self):
    """
    Initialize all of your objects here.
    Be sure to call the Framework's initializer first.
    """
    super().__init__()

    # Initialize all of the objects

  def Keyboard(self, key):
    """
    The key is from Keys.K_*
    (e.g., if key == Keys.K_z: ... )
    """
    pass

  def Step(self, settings):
    """Called upon every step.
    You should always call
     -> super().Step(settings)
    at the beginning or end of your function.

    If placed at the beginning, it will cause the actual physics step to happen first.
    If placed at the end, it will cause the physics step to happen after your code.
    """

    super().Step(settings)

    # do stuff

    # Placed after the physics step, it will draw on top of physics objects
    self.Print("*** Base your own testbeds on me! ***")

  def ShapeDestroyed(self, shape):
    """
    Callback indicating 'shape' has been destroyed.
    """
    pass

  def JointDestroyed(self, joint):
    """
    The joint passed in was removed.
    """
    pass


if __name__ == "__main__":
  framework.main(Environment)
