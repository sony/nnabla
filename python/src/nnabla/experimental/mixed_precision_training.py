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


class AutoLossScalingUpdater(object):
    '''
    TODO

    Args:
        TODO

    Example:

        .. code-block:: python

            solver = <Solver>
            loss = <Loss Variable of Network>
            updater = AutoLossScalingUpdater(solver, loss)

            # Training iteration
            for itr in range(max_iter):
                # Store input and target data to variables.
                x.d, t.d = data.next()
                # Call solver.zero_grad, loss.forward, loss.backward
                # and solver.update with dynamic loss scaling.
                updater.update()

    '''

    def __init___(self, solver, loss,
                  scale=8, scaling_factor=2, N=2000, clear_buffer=True,
                  accum_grad=1,
                  comm=None,
                  grads=[]):
        self.solver = solver
        self.loss = loss
        self.scale = scale
        self.N = N
        self.clear_buffer = clear_buffer
        self.accum_grad = accum_grad
        self.scaling_factor = scaling_factor
        self.comm = comm
        self.grads = grads
        self.counter = 0

    def update(self):
        # Initialize gradients.
        self.zero_grad()

        # Forward and backward
        for _ in range(self.accum_grad):
            # forward
            self.loss.forward(clear_no_need_grad=self.clear_buffer)

            # backward with scale
            self.loss.backward(self.scale, clear_buffer=self.clear_buffer)

        # AllReduce
        if self.comm and len(self.grads) != 0:
            self.comm.all_reduce(self.grads, division=False, inplace=False)

        # Check Inf/NaN in grads
        if self.solver.check_inf_or_nan_grad():
            self.scale /= self.scaling_factor
            self.counter = 0
            # Recursively call udpate function until no inf nor nan.
            return self.update()

        # Rescale grads
        self.solver.scale_grad(1. / self.scale)

        # Do some graident clipping, etc.

        # Update
        self.solver.update()
        if self.counter > self.N:
            self.scale *= self.scaling_factor
            self.conter = 0
        self.counter += 1
