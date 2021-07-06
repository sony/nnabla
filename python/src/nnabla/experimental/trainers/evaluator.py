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


class Evaluator(object):
    '''Evaluator

    Args: 
        vroot (:obj:`~nnabla.Variable`): Root varible of the evaluation graph.
        data_feeder (callable :obj:`object`, function, or lambda): Data feeder.
        val_iter (:obj:`int`, optional): Iterations for evaluation.
        callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before the evaluator.evalute.
        callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after the evaluator.evalute. 
        clear_buffer (:obj:`bool`, optional): Clears the no longer referenced variables during backpropagation to save memory.
        comm (:obj:`nnabla.communicators.Communicator`, optional): Communicator when to do distributed training. Default is :obj:`None`.

    Example: 

        .. code-block:: python

            from nnabla.experimental.trainers import Evaluator

            # Evaluator
            def vdata_feeder():
                ...
            def eval_callback_on_finish(i, ve):
                ...
            evaluator = Evaluator(verror, 
                                  data_feeder=vdata_feeder, 
                                  val_iter=<val_iter>, 
                                  callback_on_finish=eval_callback_on_finish)
    '''

    def _force_to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def __init__(self, vroot=None, data_feeder=None, val_iter=None,
                 callback_on_start=lambda i, v: True,
                 callback_on_finish=lambda i, v: True,
                 clear_buffer=True,
                 comm=None):
        self.vroot = self._force_to_list(vroot)
        self.data_feeder = data_feeder
        self.val_iter = val_iter
        self.callback_on_start = self._force_to_list(callback_on_start)
        self.callback_on_finish = self._force_to_list(callback_on_finish)
        self.clear_buffer = clear_buffer
        self.comm = comm

        assert len(self.vroot) == len(self.callback_on_finish)

    def evaluate(self, i):
        # Callback on start
        v_list = [0.0] * len(self.vroot)
        for v, callback in zip(v_list, self.callback_on_start):
            callback(i, v / self.val_iter)
        # Evaluation loop
        for _ in range(self.val_iter):
            # feed data
            self.data_feeder()
            # forwards
            for j, e in enumerate(zip(v_list, self.vroot)):
                v, vroot = e
                vroot.forward(clear_buffer=self.clear_buffer)
                v_list[j] += vroot.d  # accumulate
        # Callback on finish
        for v, callback in zip(v_list, self.callback_on_finish):
            callback(i, v / self.val_iter)
