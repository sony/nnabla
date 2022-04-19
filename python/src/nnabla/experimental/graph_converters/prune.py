# Copyright 2022 Sony Group Corporation.
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
import nnabla as nn
from nnabla.parameter import get_parameter_or_create
from .graph_converter import FunctionModifier
import numpy as np


class PruningModifier(FunctionModifier):
    """
    Use `PruningModifier` to prune the small weight value to 0.
    The pruning is channel-wise. Using the channel-wise L2 norm to represent the degree of sparsity.
    If the L2 norm less than the threshold provided, all the value of this channel will be set to 0.

    Supported pruning functions: `Convolution`, `Deconvolution`, `DepthwiseConvolution`, 'DepthwiseDeconvolution', 'Affine'

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.PruningModifier(pruning_threshold=0.1)]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)
    """

    def __init__(self, pruning_threshold, functions_to_prune=(
            'Convolution', 'Deconvolution', 'DepthwiseConvolution', 'DepthwiseDeconvolution', 'Affine'),
                 channel_last=False):
        """
        Args:
          pruning_threshold (float): Threshold of the L2 norm.
          functions_to_prune (list): Functions to pruning.
          channel_last (bool): If True, the data formnat of network is considered to be NHWC.

        """
        super(PruningModifier, self).__init__()
        self._channle_last = channel_last
        self._pruning_threshold = pruning_threshold

        self._default_fct_set = {
            'Affine': F.affine,
            'Convolution': F.convolution,
            'Deconvolution': F.deconvolution,
            'DepthwiseConvolution': F.depthwise_convolution,
            'DepthwiseDeconvolution': F.depthwise_deconvolution
        }

        self._fct_set = {}
        for function in functions_to_prune:
            if function in self._default_fct_set:
                self._fct_set[function] = self._default_fct_set[function]

        self.output_channel_axis = {
            'Affine': 1,
            'Convolution': 0,
            'Deconvolution': {
                True: -1,
                False: 1
            },
            'DepthwiseConvolution': 0,
            'DepthwiseDeconvolution': 0,
        }

    def calculate_axis(self, f):
        function_type = f.info.type_name
        if isinstance(self.output_channel_axis[function_type], int):
            return self.output_channel_axis[function_type]
        else:
            return self.output_channel_axis[function_type][self._channle_last]

    def modify(self, f, inputs):
        if f.info.type_name not in self._fct_set:
            return

        # Prune the weight
        x, w = inputs[:2]
        b = None
        if len(inputs) == 3:
            b = inputs[2]
        output_channel = self.calculate_axis(f)
        shape = list(range(w.ndim))
        shape.pop(output_channel)
        l2_norm_per_channel = np.sum(
            w.d ** 2, axis=tuple(shape), keepdims=True)
        mask = l2_norm_per_channel > self._pruning_threshold

        scope = self.get_parameter_scope(w)
        w_pruned, b_pruned = None, None
        with nn.parameter_scope(scope):
            w_data = w.d * mask
            w_pruned = get_parameter_or_create(
                'w-pruned', w.shape, w_data, True, True)
            if b is not None:
                b_data = b.d * mask.reshape((-1,))
                b_pruned = get_parameter_or_create(
                    'b-pruned', b_data.shape, b_data, True, True)
        h = self._fct_set[f.info.type_name](
            x, w_pruned, b_pruned, **f.info.args)
        return h
