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

import math


class RefCosine(object):

    def __init__(self, init_lr, max_iter):
        self.init_lr = init_lr
        self.max_iter = max_iter

    def get_learning_rate(self, iter):
        return self.init_lr * ((math.cos(iter * 1.0 / self.max_iter * math.pi) + 1.0) * 0.5)


@pytest.mark.parametrize("init_lr", [0.1, 0.01])
@pytest.mark.parametrize("max_iter", [1000, 10000])
def test_cosine_scheduler(init_lr, max_iter):
    scheduler_tester(
        lrs.CosineScheduler, RefCosine, max_iter, [init_lr, max_iter])
