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

import pytest
from learning_rate_scheduler_test_utils import scheduler_tester
import nnabla.utils.learning_rate_scheduler as lrs


class RefStep(object):

    def __init__(self, init_lr, gamma, iter_steps):
        self.init_lr = init_lr
        self.gamma = gamma
        self.iter_steps = iter_steps

    def get_learning_rate(self, iter):
        lr = self.init_lr
        for step in self.iter_steps:
            if iter >= step:
                lr *= self. gamma
        return lr


@pytest.mark.parametrize("init_lr", [0.1, 0.01])
@pytest.mark.parametrize("max_iter", [1000, 10000])
@pytest.mark.parametrize("gamma", [0.9, 0.1])
@pytest.mark.parametrize("iter_steps", [[500], [300, 600, 900]])
def test_step_scheduler(init_lr, max_iter, gamma, iter_steps):
    scheduler_tester(
        lrs.StepScheduler, RefStep, max_iter, [init_lr, gamma, iter_steps])
