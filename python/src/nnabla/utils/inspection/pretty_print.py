# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import absolute_import

import nnabla as nn
import numpy as np


class PrettyPrinter():
    """
    Pretty printer to print the graph structure used with the `visit` method of a Variable.

    Attributes:
        functions (list of dict): List of functions of which element is the dictionary.
            The (key, value) pair is the (`name`, function name), (`inputs`, list of input variables),
            and (`outputs`, list of output variables) of a function.
    """

    def __init__(self, summary=False, hidden=False):
        """
        Args:
            summary (bool): Print statictis of a intermediate variable.
            hidden (bool): Store the intermediate input and output variables if True.
        """
        self._summary = summary
        self._hidden = hidden
        self.functions = []

    def get_scope_name(self, x):
        params = nn.get_parameters()
        values = list(params.values())
        keys = list(params.keys())
        if x in values:
            idx = values.index(x)
            scope = "/".join(keys[idx].split("/")[:-1])
        else:
            scope = None
        return scope

    def __call__(self, f):
        scope = self.get_scope_name(f.inputs[1]) if len(f.inputs) > 1 else None
        name = "{}/{}({})".format(scope, f.name, f.info.type_name) if scope else \
            "{}({})".format(f.name, f.info.type_name)
        print(name)
        print("\tDepth = {}".format(f.rank))
        print("\tArgs:", ["{}={}".format(k, v)
                          for k, v in f.info.args.items()])
        print("\tInputs:", [i.shape for i in f.inputs])
        print("\tOutputs:", [o.shape for o in f.outputs])
        print("\tBackward Inputs:", [i.need_grad for i in f.inputs])
        if self._summary:
            print("\tInput Data:")
            print("\t\tMed: ", [np.median(i.d) for i in f.inputs])
            print("\t\tAve: ", [np.mean(i.d) for i in f.inputs])
            print("\t\tStd: ", [np.std(i.d) for i in f.inputs])
            print("\t\tMin: ", [np.min(i.d) for i in f.inputs])
            print("\t\tMax: ", [np.max(i.d) for i in f.inputs])
            print("\tOutput Data:")
            print("\t\tMed: ", [np.median(i.d) for i in f.outputs])
            print("\t\tAve: ", [np.mean(i.d) for i in f.outputs])
            print("\t\tStd: ", [np.std(i.d) for i in f.outputs])
            print("\t\tMin: ", [np.min(i.d) for i in f.outputs])
            print("\t\tMax: ", [np.max(i.d) for i in f.outputs])

            print("\tInput Grads:")
            print("\t\tMed: ", [np.median(i.g) for i in f.inputs])
            print("\t\tAve: ", [np.mean(i.g) for i in f.inputs])
            print("\t\tStd: ", [np.std(i.g) for i in f.inputs])
            print("\t\tMin: ", [np.min(i.g) for i in f.inputs])
            print("\t\tMax: ", [np.max(i.g) for i in f.inputs])
            print("\tOutput Grads:")
            print("\t\tMed: ", [np.median(i.g) for i in f.outputs])
            print("\t\tAve: ", [np.mean(i.g) for i in f.outputs])
            print("\t\tStd: ", [np.std(i.g) for i in f.outputs])
            print("\t\tMin: ", [np.min(i.g) for i in f.outputs])
            print("\t\tMax: ", [np.max(i.g) for i in f.outputs])

        if self._hidden:
            h = dict(name=name,
                     inputs=[i for i in f.inputs],
                     outputs=[o for o in f.outputs])
            self.functions.append(h)


def pprint(v, forward=False, backward=False, summary=False, hidden=False, printer=False):
    """
    Pretty print information of a graph from a root variable `v`.

    Note that in order to print the summary statistics, this function stores, i.e., does not reuse 
    the intermediate buffers of a computation graph, increasing the memory usage
    if either the forward or backward is True.

    Args:
        v (:obj:`nnabla.Variable`): Root variable.
        forward (bool): Call the forward method of a variable `v`.
        backward (bool): Call the backward method of a variable `v`.
        summary (bool): Print statictis of a intermediate variable.
        hidden (bool): Store the intermediate input and output variables if True.
        printer (bool): Return the printer object if True.

    Example:

    .. code-block:: python

        pred = Model(...)

        from nnabla.utils.inspection import pprint

        pprint(pred, summary=True, forward=True, backward=True)


    """
    v.forward() if forward else None
    v.backward() if backward else None
    pprinter = PrettyPrinter(summary, hidden)
    v.visit(pprinter)
    return pprinter if printer else None
