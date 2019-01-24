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
from __future__ import absolute_import
import nnabla as nn
from nnabla.utils.nnp_graph import NnpNetworkPass

from nnabla import logger

from .base import ImageNetBase


class MobileNetV2(ImageNetBase):
    '''
    MobileNetV2 architecture.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation..
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:
        * `Sandelr et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks.
          <https://arxiv.org/abs/1801.04381>`_

    '''

    def __init__(self):
        self._load_nnp('MobileNet-v2.nnp', 'MobileNet-v2/MobileNet-v2.nnp')

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        # Use up to
        set_use_up_to = set(
            ['classifier', 'pool', 'lastconv', 'lastconv+relu'])
        assert use_up_to in set_use_up_to, "use_up_to must be chosen from {}. Given {}.".format(
            set_use_up_to, use_up_to)

        fix_parameters = not training
        bn_batch_stat = training
        default_shape = (1,) + self.input_shape
        if input_var is None:
            input_var = nn.Variable(default_shape)
        assert input_var.ndim == 4, "input_var must be 4 dimensions. Given {}.".format(
            input_var.ndim)
        assert input_var.shape[1] == 3, "input_var.shape[1] must be 3 (RGB). Given {}.".format(
            input_var.shape[1])

        callback = NnpNetworkPass(verbose)
        callback.drop_function('ImageAugmentationX')
        callback.set_variable('ImageAugmentationX', input_var)
        if force_global_pooling:
            callback.force_average_pooling_global('AveragePooling')
        elif check_global_pooling:
            callback.check_average_pooling_global('AveragePooling')
        callback.set_batch_normalization_batch_stat_all(bn_batch_stat)

        if use_up_to == 'classifier':
            callback.use_up_to('Reshape')
        elif use_up_to == 'pool':
            callback.use_up_to('AveragePooling')
        elif use_up_to == 'lastconv':
            callback.use_up_to('BatchNormalization_2')
        elif use_up_to == 'lastconv+relu':
            callback.use_up_to('ReLU_2')
        else:
            raise NotImplementedError()
        if fix_parameters:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]
