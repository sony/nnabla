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

from nnabla.testing import assert_allclose


def scheduler_tester(scheduler, ref_scheduler, max_iter, scheduler_args=[], atol=1e-6):
    # Create scheduler
    s = scheduler(*scheduler_args)
    ref_s = ref_scheduler(*scheduler_args)

    # Check learning rate
    lr = [s.get_learning_rate(iter) for iter in range(max_iter)]
    ref_lr = [ref_s.get_learning_rate(iter) for iter in range(max_iter)]
    assert_allclose(lr, ref_lr, atol=atol)
