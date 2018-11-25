#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py Humanoid-v1 --num_rollouts 20 --render

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import load_policy

from rl.core.algs import util


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument("--max_timesteps", type=int)
  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert roll outs')
  args = parser.parse_args()

  print('loading and building expert policy')
  expert_policy_file = os.path.join('experts', args.envname + '.pkl')
  policy_fn = load_policy.load_policy(expert_policy_file)
  print('loaded and built')

  with tf.Session():
    tf_util.initialize()

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
      print('iter', i)
      obs = env.reset()
      done = False
      totalr = 0.
      steps = 0
      while not done:
        action = policy_fn(obs[None, :])
        action = action.squeeze()
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if args.render:
          env.render()
        if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
        if steps >= max_steps:
          break
      returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    # Pickle the expert_data.
    rollout_dir = os.path.join('expert_rollouts', args.envname)
    os.makedirs(rollout_dir, exist_ok=True)
    rollout_path = util.get_next_filename(
        rollout_dir,
        prefix='n' + str(args.num_rollouts) + '_',
        extension='.pkl')
    with open(rollout_path, 'wb') as fd:
      pickle.dump(expert_data, fd)

if __name__ == '__main__':
  main()
