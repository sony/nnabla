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
from explorer import GreedyExplorer
from output_path import OutputPath

from nnabla.ext_utils import get_extension_context
import nnabla as nn

import os


def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gym-env', '-g', default='BreakoutNoFrameskip-v4')
    p.add_argument('--nnp', '-m', default=None)
    p.add_argument('--nnp_dir', '-p', default=None)
    p.add_argument('--num_frames', '-f', type=int, default=4)
    p.add_argument('--no-render', '-n', action='store_true', default=True)
    p.add_argument('--extension', '-e', default='cpu')
    p.add_argument('--device-id', '-d', default='0')
    p.add_argument('--log_path', '-l', default='./tmp.output')

    return p.parse_args()


def main():

    args = get_args()

    nn.set_default_context(get_extension_context(
        args.extension, device_id=args.device_id))

    from atari_utils import make_atari_deepmind
    env = make_atari_deepmind(args.gym_env, valid=True)
    print('Observation:', env.observation_space)
    print('Action:', env.action_space)
    obs_sampler = ObsSampler(args.num_frames)
    val_replay_memory = ReplayMemory(env.observation_space.shape,
                                     env.action_space.shape, max_memory=args.num_frames)

    # for one file
    explorer = GreedyExplorer(
        env.action_space.n, use_nnp=True, nnp_file=args.nnp, name='qnet')
    validator = Validator(env, val_replay_memory, explorer, obs_sampler,
                          num_episodes=30, clip_episode_step=True,
                          render=not args.no_render)

    mean_reward = validator.step()
    with open(os.path.join(args.log_path, 'mean_reward.txt'), 'a') as f:
        print("{} {}".format(args.gym_env, str(mean_reward)), file=f)


if __name__ == '__main__':
    main()
