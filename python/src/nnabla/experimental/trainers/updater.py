# Copyright 2019,2020,2021 Sony Corporation.
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


class Updater(object):
    '''Updater

    Args:
        solver (:obj:`nnabla.solvers.Solver`): Solver object. E.g., Momentum or Adam.
        loss (:obj:`nnabla.Variable`): Loss variable from which the forward and the backward is called.
        data_feeder (callable :obj:`object`, function, or lambda): Data feeder.
        forward_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before forward function.
        forward_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after forward function.
        backward_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before backward function.
        backward_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after backward function.
        comm_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before comm.all_reduce.
        comm_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after comm.all_reduce.
        update_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before update function.
        update_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after update function.
        clear_buffer (:obj:`bool`, optional): Clears the no longer referenced variables during backpropagation to save memory.
        accum_grad (:obj:`int`, optional): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called. Default is 1.
        comm (:obj:`nnabla.communicators.Communicator`, optional): Communicator when to do distributed training. Default is :obj:`None`.
        grads (:obj:`list` of :obj:`nnabla.NdArray`, optional): The list of gradients to be exchanged when to do distributed training. Default is the empty :obj:`list`.

    Example:

        .. code-block:: python

            from nnabla.experimental.trainers import Updater

            solver = <Solver>
            loss = <Loss Variable of Network>

            def tdata_feeder():
                ...
            def update_callback_on_finish(i):
                ...
            updater = Updater(solver, loss, tdata_feeder, updater_callback_on_finish)

            # Training iteration
            for itr in range(<max_iter>):
                updater.update()
    '''

    def _force_to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def __init__(self, solver=None, loss=None,
                 data_feeder=lambda: True,
                 forward_callback_on_start=lambda i: True,
                 forward_callback_on_finish=lambda i: True,
                 backward_callback_on_start=lambda i: True,
                 backward_callback_on_finish=lambda i: True,
                 comm_callback_on_start=lambda i: True,
                 comm_callback_on_finish=lambda i: True,
                 update_callback_on_start=lambda i: True,
                 update_callback_on_finish=lambda i: True,
                 clear_buffer=True,
                 accum_grad=1,
                 comm=None,
                 grads=[]):
        self.solver = solver
        self.loss = loss
        self.data_feeder = data_feeder
        self.forward_callback_on_start = self._force_to_list(
            forward_callback_on_start)
        self.forward_callback_on_finish = self._force_to_list(
            forward_callback_on_finish)
        self.backward_callback_on_start = self._force_to_list(
            backward_callback_on_start)
        self.backward_callback_on_finish = self._force_to_list(
            backward_callback_on_finish)
        self.comm_callback_on_start = self._force_to_list(
            comm_callback_on_start)
        self.comm_callback_on_finish = self._force_to_list(
            comm_callback_on_finish)
        self.update_callback_on_start = self._force_to_list(
            update_callback_on_start)
        self.update_callback_on_finish = self._force_to_list(
            update_callback_on_finish)
        self.clear_buffer = clear_buffer
        self.accum_grad = accum_grad
        self.comm = comm
        self.grads = grads

    def update(self, i):
        """Monolithic update method.

        This method calls the following methods with the dynamic loss scaling.

        1. solver.zerograd
        2. feed data
        3. loss.forward
        4. loss.backward
        5. comm.all_reduce (if it is specified)
        6. solver.update

        """
        # Initialize gradients
        self.solver.zero_grad()

        # Forward and backward
        for _ in range(self.accum_grad):
            # feed data
            self.data_feeder()

            # forward
            for callback in self.forward_callback_on_finish:
                callback(i)
            self.loss.forward(clear_no_need_grad=self.clear_buffer)
            for callback in self.forward_callback_on_finish:
                callback(i)

            # backward
            for callback in self.backward_callback_on_start:
                callback(i)
            self.loss.backward(clear_buffer=self.clear_buffer)
            for callback in self.backward_callback_on_finish:
                callback(i)

        # AllReduce
        if self.comm and len(grads) != 0:
            for callback in self.comm_callback_on_start:
                callback(i)
            self.comm.all_reduce(self.grads, division=False, inplace=False)
            for callback in self.comm_callback_on_finish:
                callback(i)

        # Update
        for callback in self.update_callback_on_start:
            callback(i)
        self.solver.update()
        for callback in self.update_callback_on_finish:
            callback(i)
