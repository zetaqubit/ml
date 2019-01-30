from __future__ import division
from __future__ import print_function

import re

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display
#from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from PIL import Image, ImageDraw, ImageFont

def display_mat_as_gif(matrices):
  patch = plt.matshow(matrices[0])
  _display_as_gif(patch, matrices)


def display_frames_as_gif(frames):
  patch = plt.imshow(frames[0])
  _display_as_gif(patch, frames)

def _display_as_gif(patch, sequence):
  plt.axis('off')

  def animate(i):
    patch.set_data(sequence[i])

  anim = animation.FuncAnimation(plt.gcf(), animate,
      frames=len(sequence), interval=50)
  #display(display_animation(anim, default_mode='loop'))
  display(anim.to_jshtml())


def text_to_array(text):
  image = Image.new('RGB', (200, 200), (255, 255, 255))
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype(
    '/usr/share/fonts/TTF/DejaVuSansMono.ttf', 24)
  text_no_color = re.sub('\x1b\[.*?m(\w)\x1b\[.*?m',
                         lambda m: m.group(1).lower(), text)
  draw.text((0, 0), text_no_color, (0, 0, 0), font=font)
  return np.array(image)
