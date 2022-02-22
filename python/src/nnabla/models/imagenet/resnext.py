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


class ResNeXt(ImageNetBase):

    """
    ResNeXt architectures for 50 and 101 of number of layers.

    Args:
        num_layers (int): Number of layers chosen from 50 and 101.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.


    References:
        * `Xie, Girshick. et al., Aggregated Residual Transformations for Deep Neural Networks.
          <https://arxiv.org/abs/1611.05431>`_
        * `ResNeXt: Aggregated Residual Transformations for Deep Neural Networks on GitHub <https://github.com/facebookresearch/ResNeXt>`_
    """

    _KEY_VARIABLE = {
        'classifier': 'Affine',
        'pool': 'AveragePooling',
        'lastconv': 'Add2_7_RepeatStart_4[1]',
        'lastconv+relu': 'ReLU_25_RepeatStart_4[1]',
        }

    def __init__(self, num_layers=50):

        # Check validity of num_layers
        set_num_layers = set((50, 101))
        assert num_layers in set_num_layers, "num_layers must be chosen from {}".format(
            set_num_layers)
        self.num_layers = num_layers

        # Load nnp
        self._load_nnp('ResNeXt-{}.nnp'.format(num_layers),
                       'ResNeXt-{0}/ResNeXt-{0}.nnp'.format(num_layers))

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('InputX', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, 'AveragePooling')
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]


class ResNeXt50(ResNeXt):
    """ResNeXt50
        An alias of :obj:`ResNeXt` `(50)`.
    """

    def __init__(self):
        super(ResNeXt50, self).__init__(50)


class ResNeXt101(ResNeXt):
    """ResNeXt101
        An alias of :obj:`ResNeXt` `(101)`.
    """

    def __init__(self):
        super(ResNeXt101, self).__init__(101)
