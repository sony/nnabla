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


class SqueezeNet(ImageNetBase):
    """
    SqueezeNet model for architecture-v1.0 and v1.1 .

    Args:
        version (str): Version chosen from 'v1.0' and 'v1.1'.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'lastconv'``: The input of the final global average pooling without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:

        * `Iandola, Forrest N. et al., SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
          <https://arxiv.org/abs/1602.07360>`_
        * `Iandola, Forrest N. et al., SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size.
          <https://arxiv.org/abs/1602.07360v1>`_
        * `DeepScale/SqueezeNet on GitHub <https://github.com/DeepScale/SqueezeNet>`_

    """
    _KEY_VARIABLE = {
        'classifier': '{prefix}Reshape',
        'pool': '{prefix}AveragePooling',
        'lastconv': '{prefix}Convolution_2',
        'lastconv+relu': '{prefix}ReLU_2',
    }

    def __init__(self, version='v1.1'):

        # Check versions
        assert version in (
            'v1.0', 'v1.1'), "version must be chosen from {'v1.0', 'v1.1'}"
        self.version = version
        self._prefix = ''
        if version == 'v1.1':
            self._prefix = 'SqueezeNet/'

        # Load nnp
        self._load_nnp('SqueezeNet-{}.nnp'.format(version[1:]),
                       'SqueezeNet-{0}/SqueezeNet-{0}.nnp'.format(version[1:]))

    def _input_shape(self):
        return (3, 227, 227)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False,
                 force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0):

        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('TrainingInput', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, '{}AveragePooling'.format(self._prefix))
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback, prefix=self._prefix)
        if not training:
            callback.remove_and_rewire('{}Dropout'.format(self._prefix))
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]


class SqueezeNetV10(SqueezeNet):
    """SquezeNetV10
        An alias of :obj:`SqueezeNet` `('v1.0')`.
    """

    def __init__(self):
        super(SqueezeNetV10, self).__init__('v1.0')


class SqueezeNetV11(SqueezeNet):
    """SquezeNetV11
        An alias of :obj:`SqueezeNet` `('v1.1')`.
    """

    def __init__(self):
        super(SqueezeNetV11, self).__init__('v1.1')
