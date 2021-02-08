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

import numpy as np

from random cimport _set_seed

# Parameter random number generator
pseed = 313
prng = np.random.RandomState(pseed)

def seed(seed_value):
    assert isinstance(seed_value, int)
    global pseed
    global prng
    pseed = seed_value
    prng = np.random.RandomState(seed_value)
    _set_seed(seed_value)
