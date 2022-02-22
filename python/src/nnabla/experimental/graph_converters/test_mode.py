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


class TestModeModifier(object):
    """
    This converter combines BatNormBatchStateModifier and RemoveFunctionModifer.
    It changes `batch_stat` to `False`.
    Supported functions: `BatchNormalization`, `FusedBatchNormalization`, `SyncBatchNormalization`.

    Functions that specified `rm_funcs` will be removed from a graph.

    Args:
        rm_funcs (list of :obj:`str`): list of function name

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.TestModeModifier(rm_funcs=['MulScalar'])]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __new__(self, rm_funcs=[]):
        from .batch_norm_batchstat import BatchNormBatchStatModifier
        from .remove_function import RemoveFunctionModifier
        return [BatchNormBatchStatModifier(), RemoveFunctionModifier(rm_funcs=rm_funcs)]
