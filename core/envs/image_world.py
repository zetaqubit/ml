"""A large image Gym environment.

A Gym environment where the world is a large image, observable to the agent
only through a small window. The agent chooses the spatial location (x, y) and
the zoom (z) at which to place the window. Zooming in provides a more detailed
but also more local view.

The challenge is to integrate detailed information from multiple locations and
make an image-level prediction. For example, imagine 10 random MNIST digits
scattered across the world, and the goal is to predict the sum.

This is intended to be a generalization of the Cluttered Translated MNIST
task [1].

[1] https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
"""

import gym

class ImageWorldEnv(gym.core.Env):

  def __init__(self, images, window_dims):
    """Creates an ImageWorldEnv.

    The environment holds a set of images. On each reset, an image is randomly
    chosen to the world the agent explores.

    The actions available to the agent are:
      a) observe(x, y, z): center the window at pixel (x, y) and zoom z. The
         agent observes the image patch through the window.
         TODO: define coordinate system
         TODO: consider discrete vs continuous

      b) predict(c): predict the image-level characteristic of interest. The
         agent receives a reward based on the accuracy of its prediction. This
         concludes the episode, and the environment needs to be reset.

    :param images: set of images. numpy array shaped [n, h, w, c].
    :param window_dims: dimensions of the observation window. (height, width)
    """
    pass

  def step(self, action):
    pass


