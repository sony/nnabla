# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021,2023 Sony Group Corporation.
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
    '''
    Updater class definition, common for all training process.

    In earlier design of nnabla.experimental.trainers.Updater, the number of times that update function is called is passed to the callback functions.
    This number is calculated in nnabla.experimental.trainers.Trainer.train() with epoch number and iteration number.
    From this value, callback functions require iteration number per epoch to derive back the epoch number and iteration number in this epoch.
    Now, both epoch number and iteration number are passed to the callback functions, as well as iteration number per epoch.
    In case of using accumulated gradients method, the accumulated number is also passed to callback functions.
    These information enhance the flexibility of the usages of the callback functions.

    Solver and loss are not passed to Updater any more, they are defined and used in Worker.

    Clear buffer flag is maintained in Worker instead of passing through update function.

    Updater uses methods of Worker or its successors as callback functions.
    Updater instance is also declared and called in methods of Worker or its successors.

    Args:
        train_data_feeder (callable :obj:`object`, function, or lambda): Data feeder.
        solver_zero_grad (callable :obj:`object`, function, or lambda): Call solver.zero_grad.
        solver_update (callable :obj:`object`, function, or lambda): Call solver.update.
        loss_forward (callable :obj:`object`, function, or lambda, optional): Call loss.forward.
        loss_backward (callable :obj:`object`, function, or lambda, optional): Call loss.backward.
        loss_forward_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before forward function.
        loss_forward_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after forward function.
        loss_backward_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before backward function.
        loss_backward_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after backward function.
        comm_all_reduce_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before comm.all_reduce.
        comm_all_reduce (callable :obj:`object`, function, or lambda, optional): Call comm.all_reduce.
        comm_all_reduce_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after comm.all_reduce.
        solver_update_callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before update function.
        solver_update_callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after update function.
        accum_grad (:obj:`int`, optional): Number of accumulation of gradients. Update method of the `solver` is called after the `accum_grad` number of the forward and backward is called. Default is 1.
        comm (:obj:`nnabla.communicators.Communicator`, optional): Communicator when to do distributed training. Default is :obj:`None`.
        grads (:obj:`list` of :obj:`nnabla.NdArray`, optional): The list of gradients to be exchanged when to do distributed training. Default is the empty :obj:`list`.
    '''

    def _force_to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def __init__(self,
                 train_data_feeder,
                 solver_zero_grad,
                 solver_update,
                 loss_forward=lambda i, e, ag: True,
                 loss_backward=lambda i, e, ag: True,
                 loss_forward_callback_on_start=lambda i, e, ag: True,
                 loss_forward_callback_on_finish=lambda i, e, ag: True,
                 loss_backward_callback_on_start=lambda i, e, ag: True,
                 loss_backward_callback_on_finish=lambda i, e, ag: True,
                 comm_all_reduce_callback_on_start=lambda i, e: True,
                 comm_all_reduce=lambda i, e: True,
                 comm_all_reduce_callback_on_finish=lambda i, e: True,
                 solver_update_callback_on_start=lambda i, e: True,
                 solver_update_callback_on_finish=lambda i, e: True,
                 accum_grad=1,
                 comm=None,
                 grads=[]):
        self.train_data_feeder = train_data_feeder
        self.solver_zero_grad = solver_zero_grad
        self.loss_forward_callback_on_start = self._force_to_list(
            loss_forward_callback_on_start)
        self.loss_forward = loss_forward
        self.loss_forward_callback_on_finish = self._force_to_list(
            loss_forward_callback_on_finish)
        self.loss_backward_callback_on_start = self._force_to_list(
            loss_backward_callback_on_start)
        self.loss_backward = loss_backward
        self.loss_backward_callback_on_finish = self._force_to_list(
            loss_backward_callback_on_finish)
        self.comm_all_reduce_callback_on_start = self._force_to_list(
            comm_all_reduce_callback_on_start)
        self.comm_all_reduce = comm_all_reduce
        self.comm_all_reduce_callback_on_finish = self._force_to_list(
            comm_all_reduce_callback_on_finish)
        self.solver_update_callback_on_start = self._force_to_list(
            solver_update_callback_on_start)
        self.solver_update = solver_update
        self.solver_update_callback_on_finish = self._force_to_list(
            solver_update_callback_on_finish)
        self.accum_grad = accum_grad
        self.comm = comm
        self.grads = grads

    def update(self, i, e):
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
        self.solver_zero_grad(i, e)

        # Forward and backward
        for ag in range(self.accum_grad):
            # feed data
            self.train_data_feeder(i, e, ag)

            # forward
            for callback in self.loss_forward_callback_on_start:
                callback(i, e, ag)
            self.loss_forward(i, e, ag)
            for callback in self.loss_forward_callback_on_finish:
                callback(i, e, ag)

            # backward
            for callback in self.loss_backward_callback_on_start:
                callback(i, e, ag)
            self.loss_backward(i, e, ag)
            for callback in self.loss_backward_callback_on_finish:
                callback(i, e, ag)

        # AllReduce
        if self.comm and len(self.grads) != 0:
            for callback in self.comm_all_reduce_callback_on_start:
                callback(i, e)
            self.comm_all_reduce(i, e)
            for callback in self.comm_all_reduce_callback_on_finish:
                callback(i, e)

        # Update
        for callback in self.solver_update_callback_on_start:
            callback(i, e)
        self.solver_update(i, e)
        for callback in self.solver_update_callback_on_finish:
            callback(i, e)
