# Copyright 2019,2020,2021 Sony Corporation.
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

from nnabla.utils.nnp_graph import NnpNetworkPass

from .base import ImageNetBase


class DenseNet(ImageNetBase):

    """

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The output from last denseblock.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:
        * `Huang et al., Densely Connected Convolutional Networks.
          <https://arxiv.org/abs/1608.06993>`_

    """

    _KEY_VARIABLE = {
        'classifier': 'DenseNet/Affine',
        'pool': 'DenseNet/AveragePooling_4',
        'lastconv': 'DenseNet/BatchNormalization_5',
        'lastconv+relu': 'DenseNet/ReLU_5',
        }

    def __init__(self):

        # Load nnp
        self._load_nnp('DenseNet-161.nnp', 'DenseNet-161/DenseNet-161.nnp')

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)
        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('InputX', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, 'DenseNet/AveragePooling_4')
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Train', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]
