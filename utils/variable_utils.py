# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

import nnabla as nn


def get_params_startswith(str):
    return {k: v for k, v in nn.get_parameters().items() if k.startswith(str)}


def set_persistent_all(*variables):
    for var in variables:
        if var is None:
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        var.persistent = True


def set_need_grad_all(*variables, need_grad):
    assert isinstance(need_grad, bool)

    for var in variables:
        if var is None:
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        var.need_grad = need_grad


def get_unlinked_all(*variables):
    ret = []
    for var in variables:
        if var is None:
            ret.append(None)
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        ret.append(var.get_unlinked_variable())

    return ret


def zero_grads_all(*variables):
    for var in variables:
        if var is None:
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        var.grad.zero()


def fill_all(*variables, value=0):
    for var in variables:
        if var is None:
            continue

        if not isinstance(var, nn.Variable):
            raise ValueError("all variables must be nn.Variable")

        var.data.fill(value)
