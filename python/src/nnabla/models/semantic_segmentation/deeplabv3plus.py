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

from .base import SemanticSegmentation


class DeepLabV3plus(SemanticSegmentation):

    '''
    DeepLabV3+.

    Args:
        dataset(str): Specify a training dataset name from 'voc' or 'voc-coco'.
        output_stride(int): DeepLabV3 uses atrous (a.k.a. dilated) convolutions. The atrous rate depends on the output stride. the output stride has 
                     to be selected from 8 or 16. Default is 8. If the output_stride is 8 the atrous rate will be [12,24,36] 
                     and if the output_stride is 16 the atrous rate will be [6,12,18].

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'segmentation'`` (default): The output of the final layer.
    * ``'lastconv'``: The output from last Convolution.
    * ``'lastconv+relu'``: Network up to ``'lastconv'`` followed by ReLU activation.

    References:
        * `Chen et al., Rethinking Atrous Convolution for Semantic Image Segmentation.
          <https://arxiv.org/abs/1706.05587>`_

    '''

    _KEY_VARIABLE = {
        'segmentation': 'y',
        'lastconv': 'BatchNormalization_146_Output',
        'lastconv+relu': 'ReLU_82_Output',
        }

    def __init__(self, dataset='voc', output_stride=16):

        # Check validity of num_layers
        assert dataset in ['voc', 'voc-coco'],\
            'dataset must be chosen from ["voc", "voc-coco"].'
        assert output_stride in [16, 8],\
            'dataset must be chosen from [16, 8].'
        # Load nnp
        self._dataset_name = dataset
        self._output_stride = output_stride
        if self._dataset_name == 'voc':
            if self._output_stride == 16:
                self._load_nnp('DeepLabV3-voc-os-16.nnp',
                               'DeepLabV3-voc-os-16.nnp')
            else:
                self._load_nnp('DeepLabV3-voc-os-8.nnp',
                               'DeepLabV3-voc-os-8.nnp')
        elif self._dataset_name == "voc-coco":
            if self._output_stride == 16:
                self._load_nnp('DeepLabV3-voc-coco-os-16.nnp',
                               'DeepLabV3-voc-coco-os-16.nnp')
            else:
                self._load_nnp('DeepLabV3-voc-coco-os-8.nnp',
                               'DeepLabV3-voc-coco-os-8.nnp')

    def _input_shape(self):
        return (3, 513, 513)

    def __call__(self, input_var=None, use_from=None, use_up_to='segmentation', training=False, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)
        callback = NnpNetworkPass(verbose)
        callback.set_variable('x', input_var)

        '''
        Shape of output dimension for Interpolate and AveragePooling function depends on input image size
        and if these dimensions are taken from .nnp file, shape mis-matching will be encountered. So 
        these dimensions has been set according to input image shape in below callbacks.
        '''

        # changing kernel and stride dimension for AveragePoling
        @callback.on_generate_function_by_name('AveragePooling')
        def average_pooling_shape(f):
            s = f.inputs[0].proto.shape.dim[:]
            kernel_dim = f.proto.average_pooling_param
            kernel_dim.kernel.dim[:] = [s[2], s[3]]
            pool_stride = f.proto.average_pooling_param
            pool_stride.stride.dim[:] = [s[2], s[3]]
            return f

        # changing output size for all Interpolate functions, using the below callbacks.
        @callback.on_generate_function_by_name('Interpolate')
        def interpolate_output_shape(f):
            s = input_var.shape
            s1 = f.inputs[0].proto.shape.dim[:]
            w, h = s[2], s[3]
            for i in range(4):
                w = (w - 1) // 2 + 1
                h = (h - 1) // 2 + 1
            op_shape = f.proto.interpolate_param
            op_shape.output_size[:] = [w, h]
            return f

        @callback.on_generate_function_by_name('Interpolate_2')
        def interpolate_output_shape(f):
            s = input_var.shape
            w, h = s[2], s[3]
            for i in range(2):
                w = (w - 1) // 2 + 1
                h = (h - 1) // 2 + 1
            op_shape = f.proto.interpolate_param
            op_shape.output_size[:] = [w, h]
            return f

        @callback.on_generate_function_by_name('Interpolate_3')
        def interpolate_output_shape(f):
            s = input_var.shape
            op_shape = f.proto.interpolate_param
            op_shape.output_size[:] = [s[2], s[3]]
            return f

        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if not training:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'runtime', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]
