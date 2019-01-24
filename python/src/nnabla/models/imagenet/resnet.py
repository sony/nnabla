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


class ResNet(ImageNetBase):

    '''
    ResNet architectures for 18, 34, 50, 101, and 152 of number of layers.

    Args:
        num_layers (int): Number of layers chosen from 18, 34, 50, 101, and 152.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation..
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:
        * `He et al, Deep Residual Learning for Image Recognition.
          <https://arxiv.org/abs/1512.03385>`_

    '''

    def __init__(self, num_layers=18):

        # Check validity of num_layers
        set_num_layers = set((18, 34, 50, 101, 152))
        assert num_layers in set_num_layers, "num_layers must be chosen from {}".format(
            set_num_layers)
        self.num_layers = num_layers

        # Load nnp
        self._load_nnp('Resnet-{}.nnp'.format(num_layers),
                       'Resnet-{0}/Resnet-{0}.nnp'.format(num_layers))

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'

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

        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('InputX', input_var)
        if force_global_pooling:
            callback.force_average_pooling_global('AveragePooling')
        elif check_global_pooling:
            callback.check_average_pooling_global('AveragePooling')
        callback.set_batch_normalization_batch_stat_all(bn_batch_stat)

        if use_up_to == 'classifier':
            callback.use_up_to('Affine')
        elif use_up_to == 'pool':
            callback.use_up_to('AveragePooling')
        elif use_up_to.startswith('lastconv'):
            callback.use_up_to('Add2_7_RepeatStart_4[{}]'.format(
                0 if self.num_layers == 18 else 1))
        elif use_up_to.startswith('lastconv+relu'):
            callback.use_up_to('ReLU_25_RepeatStart_4[{}]'.format(
                0 if self.num_layers == 18 else 1))
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
