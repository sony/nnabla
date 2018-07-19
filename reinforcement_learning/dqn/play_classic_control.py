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

from sampler import ObsSampler
from replay_memory import ReplayMemory
from validator import Validator
from explorer import NnpExplorer

from nnabla.ext_utils import get_extension_context
import nnabla as nn


def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gym-env', '-g', default='CartPole-v0')
    p.add_argument('--nnp', '-m', default=None)
    p.add_argument('--num_frames', '-f', type=int, default=1)
    p.add_argument('--no-render', '-n', action='store_true', default=False)
    p.add_argument('--extension', '-e', default='cpu')
    p.add_argument('--device-id', '-d', default='0')

    return p.parse_args()


def main():

    args = get_args()

    nn.set_default_context(get_extension_context(
        args.extension, device_id=args.device_id))

    import gym
    env = gym.make(args.gym_env)
    print('Observation:', env.observation_space)
    print('Action:', env.action_space)
    obs_sampler = ObsSampler(args.num_frames)
    val_replay_memory = ReplayMemory(
        env.observation_space.shape, env.action_space.shape, max_memory=args.num_frames)
    explorer = NnpExplorer(
        args.nnp, env.action_space.n, name='qnet')
    validator = Validator(env, val_replay_memory, explorer, obs_sampler,
                          num_episodes=1, render=not args.no_render,
                          log_path=False)
    while True:
        validator.step()


if __name__ == '__main__':
    main()
