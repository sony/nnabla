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

from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_source_loader import download
from nnabla.logger import logger

import nnabla as nn

import os


def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gym-env', '-g', default='BreakoutNoFrameskip-v4')
    p.add_argument('--nnp', '-m', default=None)
    p.add_argument('--num_frames', '-f', type=int, default=4)
    p.add_argument('--no-render', '-n', action='store_true', default=False)
    p.add_argument('--extension', '-e', default='cpu')
    p.add_argument('--device-id', '-d', default='0')

    return p.parse_args()


def find_local_nnp(env_name):
    nnp_fpath = os.path.join("asset", env_name, "qnet.nnp")
    return os.path.exists(nnp_fpath)


def main():

    args = get_args()

    nn.set_default_context(get_extension_context(
        args.extension, device_id=args.device_id))

    if args.nnp is None:
        local_nnp_dir = os.path.join("asset", args.gym_env)
        local_nnp_file = os.path.join(local_nnp_dir, "qnet.nnp")

        if not find_local_nnp(args.gym_env):
            logger.info("Downloading nnp data since you didn't specify...")
            nnp_uri = os.path.join("https://nnabla.org/pretrained-models/nnp_models/examples/dqn",
                                   args.gym_env,
                                   "qnet.nnp")
            if not os.path.exists(local_nnp_dir):
                os.mkdir(local_nnp_dir)
            download(nnp_uri, output_file=local_nnp_file, open_file=False)
            logger.info("Download done!")

        args.nnp = local_nnp_file

    from atari_utils import make_atari_deepmind
    env = make_atari_deepmind(args.gym_env, valid=False)
    print('Observation:', env.observation_space)
    print('Action:', env.action_space)
    obs_sampler = ObsSampler(args.num_frames)
    val_replay_memory = ReplayMemory(env.observation_space.shape,
                                     env.action_space.shape,
                                     max_memory=args.num_frames)
    # just play greedily
    explorer = GreedyExplorer(
        env.action_space.n, use_nnp=True, nnp_file=args.nnp, name='qnet')
    validator = Validator(env, val_replay_memory, explorer, obs_sampler,
                          num_episodes=1, render=not args.no_render)
    while True:
        validator.step()


if __name__ == '__main__':
    main()
