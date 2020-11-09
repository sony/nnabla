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

import nnabla.solvers as S
from nnabla.utils.learning_rate_scheduler import BaseLearningRateScheduler


class Optimizer(object):
    """An Optimizer class.

    Args:
        weight_decay (float, optional): Weight decay (L2 penalty). Should be
            a positive value. Defaults to 0.
        max_norm (float, optional): An input scalar of float value. Should be
            a positive value. Defaults to 0.
        lr_scheduler (`BaseLearningRateScheduler`, optional): Learning rate
            scheduler. Defaults to None (no learning rate scheduler is applied).
        name (str, optional): Name of the solver. Defaults to 'Sgd'.

    Raises:
        NotImplementedError: If the solver is not supported in NNabla.
    """

    def __init__(self, weight_decay=0, max_norm=0,
                 lr_scheduler=None, name='Sgd', **kargs):

        if name not in S.__dict__:
            raise NotImplementedError(name + 'is not implemented')

        self._solver = S.__dict__[name](**kargs)
        self._weight_decay = weight_decay
        self._max_norm = max_norm
        self._lr_scheduler = lr_scheduler
        self._iter = 0  # current iter

        if lr_scheduler is not None:
            lr = self._lr_scheduler.get_learning_rate(self._iter)
            self._solver.set_learning_rate(lr)

    def set_parameters(self, params, **kargs):
        r"""Set parameters to the solver."""
        self._solver.set_parameters(params, **kargs)

    def update(self):
        r"""Update parameters."""
        if self._lr_scheduler is not None:
            lr = self._lr_scheduler.get_learning_rate(self._iter)
            self._solver.set_learning_rate(lr)

        if self._weight_decay > 0:
            self._solver.weight_decay(self._weight_decay)

        if self._max_norm > 0:
            self._solver.clip_grad_by_norm(self._max_norm)

        self._solver.update()
        self._iter += 1

    def zero_grad(self):
        r"""Make zero gradients for all current parameters."""
        self._solver.zero_grad()

    def get_parameters(self):
        r"""Get current parameters."""
        return self._solver.get_parameters()

    def get_learning_rate(self):
        r"""Get the current learning rate."""
        return self._solver.learning_rate()

    def clear_parameters(self):
        r"""Clear all the current parameters."""
        self._solver.clear_parameters()
        self._iter = 0
