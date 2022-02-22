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


class InceptionV3(ImageNetBase):
    """
    InceptionV3 architecture.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'prepool'``: The input of the final global average pooling, i.e. the output of the final inception block.

    References:
        * `Szegedy et al., Rethinking the Inception Architecture for Computer Vision.
          <https://arxiv.org/abs/1512.00567>`_
    """

    _KEY_VARIABLE = {
        'classifier': 'Affine',
        'pool': 'AveragePooling_2',
        'prepool': 'Inception_11/Concatenate',
        '_aux_classifier': 'Affine_2',
        '_include_no_aux': 'Conv_6/Convolution'
        }

    def __init__(self):
        self._load_nnp('Inception-v3.nnp', 'Inception-v3/Inception-v3.nnp')

    def _input_shape(self):
        return (3, 299, 299)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0, with_aux_tower=False):
        if not training:
            assert not with_aux_tower, "Aux Tower should be disabled when inference process."

        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentation')
        callback.set_variable('Iv3TrainInput', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, 'AveragePooling_2')
        callback.set_batch_normalization_batch_stat_all(training)
        if with_aux_tower:
            self.use_up_to('_aux_classifier', callback)
            funcs_to_drop = ("Affine_2",
                             "SoftmaxCrossEntropy_2",
                             "MulScalar_2")
        else:
            self.use_up_to('_include_no_aux', callback)
            funcs_to_drop = ("Conv_6/Convolution",
                             "Conv_6/BatchNormalization",
                             "Conv_6/ReLU",
                             "AveragePooling",
                             "Conv_7/Convolution",
                             "Conv_7/BatchNormalization",
                             "Conv_7/ReLU",
                             "Affine_2",
                             "SoftmaxCrossEntropy_2",
                             "MulScalar_2")

        callback.drop_function(*funcs_to_drop)
        if not training:
            callback.remove_and_rewire('Dropout')
            callback.fix_parameters()
        self.use_up_to(use_up_to, callback)
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'Train', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        elif with_aux_tower:
            return list(net.outputs.values())
        else:
            return list(net.outputs.values())[0]
