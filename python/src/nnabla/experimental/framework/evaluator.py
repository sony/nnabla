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


class Evaluator(object):
    '''
    Evaluator class definition, common for all evaluation process.

    In earlier design of nnabla.experimental.trainers.Evaluator, hidden assumption is each callback on finish handles one evaluator's result.
    The input vroot and callback_on_finish have to be the same in size and order.
    Now, the implementation of traversal is moved into the callback functions implemented in Worker, and values to monitor are also maintained in it.

    Communicator is removed since it is not used. We can add it back to enhance performance in future.

    Evaluator uses methods of Worker or its successors as callback functions.
    Evaluator instance is also declared and called in methods of Worker or its successors.

    Args:
        eval_data_feeder (callable :obj:`object`, function, or lambda): Callback called to get next data for evaluation.
        eval_graph_forward (callable :obj:`object`, function, or lambda): Callback called to do forward on evaluation graphs.
        callback_on_start (callable :obj:`object`, function, lambda, or list of these, optional): Callback called before the evaluator.evalute.
        callback_on_finish (callable :obj:`object`, function, lambda, or list of these, optional): Callback called after the evaluator.evalute.
        val_iter (:obj:`int`, optional): Iterations for evaluation.
    '''

    def _force_to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def __init__(self,
                 eval_data_feeder,
                 eval_graph_forward,
                 eval_callback_on_start=lambda i: True,
                 eval_callback_on_finish=lambda i: True,
                 val_iter=1):
        self.eval_data_feeder = eval_data_feeder
        self.eval_graph_forward = eval_graph_forward
        self.eval_callback_on_start = self._force_to_list(
            eval_callback_on_start)
        self.eval_callback_on_finish = self._force_to_list(
            eval_callback_on_finish)
        self.val_iter = val_iter

    def evaluate(self, e):
        # Callback on start
        for callback in self.eval_callback_on_start:
            callback(e)
        # Evaluation loop
        for vi in range(self.val_iter):
            # feed data
            self.eval_data_feeder(e, vi)
            # forwards
            self.eval_graph_forward(e, vi)
        # Callback on finish
        for callback in self.eval_callback_on_finish:
            callback(e)
