# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import os
import numpy as np

from replay_memory import ReplayMemory
from sampler import Sampler, ObsSampler
from learner import QLearner, q_cnn
from explorer import EpsilonGreedyQExplorer
from trainer import Trainer
from validator import Validator


from nnabla.ext_utils import get_extension_context
import nnabla as nn


class Experiment(object):

    def __init__(self, trainer, validator):

        self.trainer = trainer
        self.validator = validator

    def step(self):
        self.trainer.step()
        self.validator.step()


def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gym-env', '-g', default='BreakoutNoFrameskip-v4')
    p.add_argument('--num_epochs', '-E', type=int, default=10000)
    p.add_argument('--num_episodes', '-T', type=int, default=10)
    p.add_argument('--num_val_episodes', '-V', type=int, default=2)
    p.add_argument('--num_frames', '-f', type=int, default=4)
    p.add_argument('--render-train', '-r', action='store_true')
    p.add_argument('--render-val', '-v', action='store_true')
    p.add_argument('--extension', '-e', default='cpu')
    p.add_argument('--device-id', '-d', default='0')

    return p.parse_args()


def main():

    args = get_args()

    nn.set_default_context(get_extension_context(
        args.extension, device_id=args.device_id))

    # Create an atari env.
    from atari_utils import make_atari_deepmind
    env = make_atari_deepmind(args.gym_env)
    print('Observation:', env.observation_space)
    print('Action:', env.action_space)

    # 10000 * 4 frames
    replay_memory = ReplayMemory(
        env.observation_space.shape, env.action_space.shape, max_memory=40000)
    learner = QLearner(q_cnn, env.action_space.n, sync_freq=1000,
                       gamma=0.99, learning_rate=1e-4, name_q='q')
    explorer = EpsilonGreedyQExplorer(
        q_cnn, env.action_space.n, eps_start=1.0, eps_end=0.01, eps_steps=1e6, name='q')
    sampler = Sampler(args.num_frames)
    obs_sampler = ObsSampler(args.num_frames)
    trainer = Trainer(env, replay_memory, learner, sampler, explorer, obs_sampler,
                      num_episodes=args.num_episodes, train_start=10000, batch_size=32,
                      render=args.render_train)
    val_replay_memory = ReplayMemory(
        env.observation_space.shape, env.action_space.shape, max_memory=args.num_frames)
    validator = Validator(env, val_replay_memory, explorer, obs_sampler,
                          num_episodes=args.num_val_episodes, render=args.render_val)
    exper = Experiment(trainer, validator)
    for e in range(args.num_epochs):
        exper.step()


if __name__ == '__main__':
    main()
