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


class ShuffleNet(ImageNetBase):
    """
    Model for architecture ShuffleNet, ShuffleNet-0.5x and ShufffleNet-2.0x.

    Args:
        Scaling Factor (str): To customize the network to a desired complexity,  we can simply apply a scale factor on the number of channnels. This can be chosen from '10', '5' and '20'.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:
        * `Zhang. et al., ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.
          <https://arxiv.org/abs/1707.01083>`_
        * `Ma, Zhang. et al., ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.
          <https://arxiv.org/abs/1807.11164>`_
    """

    _KEY_VARIABLE = {
        'classifier': 'Affine',
        'pool': 'AveragePooling',
        'lastconv': 'Add2_3_RepeatStart_3[2]',
        'lastconv+relu': 'ReLU_8_RepeatStart_3[2]',
        }

    def __init__(self, scaling_factor=10):

        # Check validity of scaling factor
        set_scaling_factor = set((10, 5, 20))
        assert scaling_factor in set_scaling_factor, "scaling_factor must be chosen from {}".format(
            set_scaling_factor)
        self.scaling_factor = scaling_factor

        # Load nnp
        if scaling_factor == 10:
            self._load_nnp('ShuffleNet.nnp',
                           'ShuffleNet/ShuffleNet.nnp')
        elif scaling_factor == 5:
            self._load_nnp('ShuffleNet-.05x.nnp',
                           'ShuffleNet-0.5x/ShuffleNet-0.5x.nnp')
        elif scaling_factor == 20:
            self._load_nnp('ShuffleNet-2.0x.nnp',
                           'ShuffleNet-2.0x/ShuffleNet-2.0x.nnp')

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


class ShuffleNet10(ShuffleNet):
    """ShuffleNet10
        An alias of :obj:`ShuffleNet` `(10)`.
    """

    def __init__(self):
        super(ShuffleNet10, self).__init__(10)


class ShuffleNet05(ShuffleNet):
    """ShuffleNet05
        An alias of :obj:`ShuffleNet` `(5)`.
    """

    def __init__(self):
        super(ShuffleNet05, self).__init__(5)


class ShuffleNet20(ShuffleNet):
    """ShuffleNet20
        An alias of :obj:`ShuffleNet` `(20)`.
    """

    def __init__(self):
        super(ShuffleNet20, self).__init__(20)
