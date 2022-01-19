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

import nnabla.functions as F

from .graph_converter import FunctionModifier


class BatchNormBatchStatModifier(FunctionModifier):
    """
    Change `batch_stat` to `False`.
    Supported functions: `BatchNormalization`, `FusedBatchNormalization`, `SyncBatchNormalization`.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.BatchNormBatchStatModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __init__(self):
        super(BatchNormBatchStatModifier, self).__init__()

        self._fct_set = {
            'BatchNormalization': F.batch_normalization,
            'FusedBatchNormalization': F.fused_batch_normalization,
            'SyncBatchNormalization': F.sync_batch_normalization
        }

    def connect(self, fname, inputs, args):
        fct = self._fct_set[fname]
        args['batch_stat'] = False

        if 'no_scale' in args:
            del args['no_scale']
        if 'no_bias' in args:
            del args['no_bias']

        h = fct(*inputs, **args)
        return h

    def modify(self, f, inputs):
        if not f.info.type_name in self._fct_set:
            return

        h = self.connect(f.info.type_name, inputs, f.info.args)

        return h
