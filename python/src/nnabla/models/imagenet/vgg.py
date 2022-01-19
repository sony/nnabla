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


class VGG(ImageNetBase):

    """
    VGG architectures for 11, 13, 16 layers.

    Args:
        num_layers (int): Number of layers chosen from 11, 13, 16.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.
    * ``'lastfeature'``: Network up to one layer before ``'classifier'``, but without activation.

    References:
        * `Simonyan and Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition.
          <https://arxiv.org/pdf/1409.1556>`_

    """

    def __init__(self, num_layers=11):

        # Check validity of num_layers
        set_num_layers = set((11, 13, 16))
        assert num_layers in set_num_layers, "num_layers must be chosen from {}".format(
            set_num_layers)
        self.num_layers = num_layers

        # Load nnp
        self._load_nnp('VGG-{}.nnp'.format(num_layers),
                       'VGG-{0}/VGG-{0}.nnp'.format(num_layers))

        self._KEY_VARIABLE = {
            'classifier': 'VGG{}/Affine_3'.format(num_layers),
            'pool': 'VGG{}/MaxPooling_5'.format(num_layers),
            'lastconv': 'VGG16/Convolution_13' if num_layers == 16 else 'VGG{}/Convolution_12'.format(num_layers),
            'lastconv+relu': 'VGG16/ReLU_13' if num_layers == 16 else 'VGG{}/ReLU_12'.format(num_layers),
            'lastfeature': 'VGG{}/Affine_2'.format(num_layers),
            }

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False,
                 force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('TrainingInput', input_var)
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.remove_and_rewire(
                'VGG{}/Dropout_1'.format(self.num_layers))
            callback.remove_and_rewire(
                'VGG{}/Dropout_2'.format(self.num_layers))
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]


class VGG11(VGG):
    """VGG11
        An alias of :obj:`VGG` `(11)`.
    """

    def __init__(self):
        super(VGG11, self).__init__(11)


class VGG13(VGG):
    """VGG13
        An alias of :obj:`VGG` `(13)`.
    """

    def __init__(self):
        super(VGG13, self).__init__(13)


class VGG16(VGG):
    """VGG16
        An alias of :obj:`VGG` `(16)`.
    """

    def __init__(self):
        super(VGG16, self).__init__(16)
