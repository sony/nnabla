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


class RefPolynomial(object):

    def __init__(self, init_lr, max_iter, power):
        self.init_lr = init_lr
        self.max_iter = max_iter
        self.power = power

    def get_learning_rate(self, iter):
        return self.init_lr * ((1.0 - iter * 1.0 / self.max_iter) ** self.power)


@pytest.mark.parametrize("init_lr", [0.1, 0.01])
@pytest.mark.parametrize("max_iter", [1000, 10000])
@pytest.mark.parametrize("power", [0.5, 1, 2])
def test_polynomial_scheduler(init_lr, max_iter, power):
    scheduler_tester(
        lrs.PolynomialScheduler, RefPolynomial, max_iter, [init_lr, max_iter, power])
