# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import pytest
from learning_rate_scheduler_test_utils import scheduler_tester
import nnabla.utils.learning_rate_scheduler as lrs


class RefExponential(object):

    def __init__(self, init_lr, gamma, iter_interval):
        self.init_lr = init_lr
        self.gamma = gamma
        self.iter_interval = iter_interval

    def get_learning_rate(self, iter):
        return self.init_lr * (self.gamma ** (iter // self.iter_interval))


@pytest.mark.parametrize("init_lr", [0.1, 0.01])
@pytest.mark.parametrize("max_iter", [1000, 10000])
@pytest.mark.parametrize("gamma", [0.9, 0.1])
@pytest.mark.parametrize("iter_interval", [100, 1000])
def test_exponential_scheduler(init_lr, max_iter, gamma, iter_interval):
    scheduler_tester(
        lrs.ExponentialScheduler, RefExponential, max_iter, [init_lr, gamma, iter_interval])
