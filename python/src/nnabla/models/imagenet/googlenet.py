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


class GoogLeNet(ImageNetBase):
    """
    GoogLeNet model.

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'classifier'`` (default): The output of the final affine layer for classification.
    * ``'pool'``: The output of the final global average pooling.
    * ``'prepool'``: The input of the final global average pooling, i.e. the output of the final inception block.

    References:
        * `Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich: Going Deeper with Convolutions.
          <https://arxiv.org/pdf/1409.4842.pdf>`_

    """
    _KEY_VARIABLE = {
        'classifier': 'Affine_5',
        'pool': 'AveragePooling_3',
        'prepool': 'Concatenate_9',
        '_aux_classifier_1': 'Affine_2',
        '_branching_point_1': 'AveragePooling',
        '_aux_classifier_2': 'Affine_4',
        '_branching_point_2': 'AveragePooling_2'
        }

    def __init__(self):
        # Load nnp
        self._load_nnp('GoogLeNet.nnp', 'GoogLeNet/GoogLeNet.nnp')

    def _input_shape(self):
        return (3, 224, 224)

    def __call__(self, input_var=None, use_from=None, use_up_to='classifier', training=False, force_global_pooling=False, check_global_pooling=True, returns_net=False, verbose=0, with_aux_tower=False):
        if not training:
            assert not with_aux_tower, "Aux Tower should be disabled when inference process."

        input_var = self.get_input_var(input_var)

        callback = NnpNetworkPass(verbose)
        callback.remove_and_rewire('ImageAugmentationX')
        callback.set_variable('InputX', input_var)
        self.configure_global_average_pooling(
            callback, force_global_pooling, check_global_pooling, 'AveragePooling_3')
        callback.set_batch_normalization_batch_stat_all(training)
        if with_aux_tower:
            self.use_up_to('_aux_classifier_1', callback)
            funcs_to_drop1 = ("Affine_2",
                              "SoftmaxCrossEntropy",
                              "MulScalarLoss1")

            self.use_up_to('_aux_classifier_2', callback)
            funcs_to_drop2 = ("Affine_4",
                              "SoftmaxCrossEntropy_2",
                              "MulScalarLoss2")
        else:
            self.use_up_to('_branching_point_1', callback)
            funcs_to_drop1 = ("AveragePooling",
                              "Convolution_22",
                              "ReLU_22",
                              "Affine",
                              "ReLU_23",
                              "Dropout",
                              "Affine_2",
                              "SoftmaxCrossEntropy",
                              "MulScalarLoss1")

            self.use_up_to('_branching_point_2', callback)
            funcs_to_drop2 = ("AveragePooling_2",
                              "Convolution_41",
                              "ReLU_42",
                              "Affine_3",
                              "ReLU_43",
                              "Dropout_2",
                              "Affine_4",
                              "SoftmaxCrossEntropy_2",
                              "MulScalarLoss2")
        callback.drop_function(*funcs_to_drop1)
        callback.drop_function(*funcs_to_drop2)
        if not training:
            callback.remove_and_rewire('Dropout_3')
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
