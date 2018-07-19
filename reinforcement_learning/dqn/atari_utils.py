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

import gym
import numpy as np


class Squeeze(gym.ObservationWrapper):
    '''Assume wrap_deepmind with scale=True'''

    def __init__(self, env):
        from gym import spaces
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(84, 84), dtype=np.float32)

    def observation(self, observation):
        return np.squeeze(observation)


def make_atari_deepmind(rom_name):
    from external.atari_wrappers import make_atari, wrap_deepmind
    env = make_atari(rom_name)
    # framestack is handled by sampler.py
    env = wrap_deepmind(env, frame_stack=False, scale=True)
    env = Squeeze(env)
    return env
