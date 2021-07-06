# Copyright 2018,2019,2020,2021 Sony Corporation.
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


class DynamicLossScalingUpdater(object):
    '''Dynamic Loss Scaling Updater for the mixed precision training.

    Args:
        solver (:obj:`nnabla.solvers.Solver`): Solver object. E.g., Momentum or Adam.
        loss (:obj:`nnabla.Variable`): Loss variable from which the forward and the backward is called.
        data_feeder (callable :obj:`object`, function, or lambda): Data feeder
        scale (:obj:`float`): Loss scale constant. This is dynamically changing during training.
        scaling_factor (:obj:`float`): Scaling factor for the dynamic loss scaling.
        N (:obj:`int`): Interval, the number of iterations in training for increasing `loss scale` by `scaling_factor`.
        clear_buffer (:obj:`bool`): Clears the no longer referenced variables during backpropagation to save memory.
        accum_grad (:obj:`int`): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called.
        weight_decay (:obj:`float`): Decay constant. Default is `None`, not applying the weight decay.
        comm (:obj:`nnabla.communicators.Communicator`): Communicator when to do distributed training. Default is :obj:`None`.
        grads (:obj:`list` of :obj:`nnabla.NdArray`): The list of gradients to be exchanged when to do distributed training. Default is the empty :obj:`list`.

    Attributes:
        solver (:obj:`nnabla.solvers.Solver`): Solver object. E.g., Momentum or Adam.
        loss (:obj:`nnabla.Variable`): Loss variable from which the forward and the backward is called.
        data_feeder (callable :obj:`object`, function, lambda): Data feeder
        scale (:obj:`float`): Loss scale constant. This is dynamically changing during training.
        scaling_factor (:obj:`float`): Scaling factor for the dynamic loss scaling.
        N (:obj:`int`): Interval, the number of iterations in training for increasing `loss scale` by `scaling_factor`.
        clear_buffer (:obj:`bool`): Clears the no longer referenced variables during backpropagation to save memory.
        accum_grad (:obj:`int`): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called.
        weight_decay (:obj:`float`): Decay constant. Default is `None`, not applying the weight decay.
        comm (:obj:`nnabla.communicators.Communicator`): Communicator when to do distributed training.
        grads (:obj:`list` of :obj:`nnabla.NdArray`): The list of gradients to be exchanged when to do distributed training.

    Example:

        .. code-block:: python
            solver = <Solver>
            loss = <Loss Variable of Network>
            data_feeder = <DataFeeder>

            updater = DynamicLossScalingUpdater(solver, loss, data_feeder)

            # Training iteration
            for itr in range(max_iter):
                # Call solver.zero_grad, data_feeder, loss.forward, loss.backward
                # and solver.update with the dynamic loss scaling.
                updater.update()

    Reference:

        https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor

    '''

    def __init__(self, solver, loss, data_feeder=lambda x: x,
                 scale=8.0, scaling_factor=2.0, N=2000, clear_buffer=True,
                 accum_grad=1, weight_decay=None,
                 comm=None,
                 grads=[]):
        self.solver = solver
        self.loss = loss
        self.data_feeder = data_feeder
        self.scale = scale
        self.scaling_factor = scaling_factor
        self.N = N
        self.clear_buffer = clear_buffer
        self.accum_grad = accum_grad
        self.weight_decay = weight_decay
        self.comm = comm
        self.grads = grads
        self._counter = 0
        self._recursive_count = 0
        self._max_recursive_count = 100

    def update(self):
        """Monolithic update method.

        This method calls the following methods with the dynamic loss scaling.

        1. solver.zerograd
        2. feed data
        3. loss.forward
        4. loss.backward
        5. comm.all_reduce (if it is specified)
        6. solver.update

        """

        # Initialize gradients.
        self.solver.zero_grad()

        # Forward and backward
        for _ in range(self.accum_grad):
            # feed data
            self.data_feeder()

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
            self._counter = 0

            # Recursively call udpate function until no inf nor nan.
            self._recursive_count += 1
            if self._recursive_count > self._max_recursive_count:
                self._recursive_count = 0
                return  # skip
            return self.update()
        self._recursive_count = 0

        # Rescale grads
        self.solver.scale_grad(1. / self.scale)

        # Do some gradient clipping, etc.
        if self.weight_decay is not None:
            self.solver.weight_decay(self.weight_decay)

        # Update
        self.solver.update()
        if self._counter > self.N:
            self.scale *= self.scaling_factor
            self._counter = 0
        self._counter += 1
