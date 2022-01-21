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

from __future__ import absolute_import

from nnabla.utils.nnp_graph import NnpNetworkPass

from .base import ImageNetBase


class AlexNet(ImageNetBase):
    """
    AlexNet model.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final pooling layer.
    * ``'lastconv'``: The input of the final pooling layer without ReLU activation.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.
    * ``'lastfeature'``: Network up to one layer before ``'classifier'``, but without activation.

    References:
        * `Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton: ImageNet Classification with Deep ConvolutionalNeural Networks.
          <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    """
    _KEY_VARIABLE = {
        'classifier': 'TrainNet/Affine_3',
        'pool': 'TrainNet/MaxPooling_3',
        'lastconv': 'TrainNet/Convolution_5',
        'lastconv+relu': 'TrainNet/ReLU_5',
        'lastfeature': 'TrainNet/Affine_2',
        }

    def __init__(self):
        # Load nnp
        self._load_nnp('AlexNet.nnp', 'AlexNet/AlexNet.nnp')

    def _input_shape(self):
        return (3, 227, 227)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0, with_aux_tower=False):

        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('TrainingInput', input_var)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.remove_and_rewire('TrainNet/Dropout')
            callback.remove_and_rewire('TrainNet/Dropout_2')
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Training', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        else:
            return list(net.outputs.values())[0]
