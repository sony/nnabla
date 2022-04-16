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
import nnabla as nn
from nnabla.utils.nnp_graph import NnpNetworkPass

import numpy as np
from .base import ObjectDetection


class YoloV2(ObjectDetection):

    '''

    The following is a list of string that can be specified to ``use_up_to`` option in ``__call__`` method;

    * ``'detection'`` (default): The output from the last convolution (detection layer) after post-processing.
    * ``'convdetect'``: The output of last convolution without post-processing.
    * ``'lastconv'``: Network till the convolution layer+relu which comes before detection convolution layer.

    References:
        * `Joseph Redmon et al., YOLO9000: Better, Faster, Stronger.
          <https://arxiv.org/abs/1612.08242>`_

    '''

    _KEY_VARIABLE = {
        'detection': 'y',
        'convdetect': 'Reshape_3_Output',
        'lastconv': 'Reshape_3_Output',
        'Arange': 'Arange_Output',
        'Arange2': 'Arange_2_Output',
        }

    def __init__(self, dataset='voc'):

        # Check validity of num_layers
        assert dataset in ['voc', 'coco'],\
            'dataset must be chosen from ["voc", "coco"].'
        # Load nnp
        self._dataset_name = dataset
        if self._dataset_name == 'voc':
            self._load_nnp('yolov2-voc.nnp', 'yolov2-voc.nnp')
        elif self._dataset_name == "coco":
            self._load_nnp('yolov2-coco.nnp', 'yolov2-coco.nnp')

    def _input_shape(self):
        return (3, 416, 416)

    def __call__(self, input_var=None, use_from=None, use_up_to='detection', training=False, returns_net=False, verbose=0):

        assert use_from is None, 'This should not be set because it is for forward compatibility.'
        input_var = self.get_input_var(input_var)
        nnp_input_size = self.get_nnp_input_size()
        callback = NnpNetworkPass(verbose)
        callback.set_variable('x', input_var)
        callback.set_batch_normalization_batch_stat_all(training)
        self.use_up_to(use_up_to, callback)
        if use_up_to != 'detection':
            self.use_up_to('Arange', callback)
            self.use_up_to('Arange2', callback)
            funcs_to_drop = ('Reshape_3',
                             'Arange',
                             'Arange_2')
            callback.drop_function(*funcs_to_drop)
            if use_up_to == 'lastconv':
                callback.drop_function('Convolution_23')

        # Output dimension of reshape, arange, slice etc functions are taken from .nnp file.
        # These dimensions depend on the input image size with which the nnp file was created.
        # When different input image size is given to the model, these dimensions will change and therefore
        # shape of output from these functions need to be generalized whenevr they are generated.
        # The same has been done in below callbacks.

        # Reshape operation for simulating darknet reorg bug
        @callback.on_generate_function_by_name('Reshape')
        def reshape_for_darknet_reorg_bug(f):
            s = f.inputs[0].proto.shape.dim[:]
            stride = 2
            r = f.proto.reshape_param
            r.shape.dim[:] = [
                s[0], int(s[1]/stride/stride), s[2], stride, s[3], stride]
            return f

        # Reshape operation for simulating darknet reorg bug
        @callback.on_generate_function_by_name('Reshape_2')
        def reshape_for_darknet_reorg_bug(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[:] = [s[0], s[1]*s[2]*s[3]
                              * s[1]*s[2], s[4]//s[1], s[5]//s[2]]
            return f

        # Reshape operation for output variable of yolov2 function in yolov2_activate.
        @callback.on_generate_function_by_name('Reshape_3')
        def reshape_yolov2_activate(f):
            s = f.inputs[0].proto.shape.dim[:]
            anchors = 5
            r = f.proto.reshape_param
            num_class = r.shape.dim[2] - 5
            s_add = (s[0], anchors, num_class+5)+tuple(s[2:])
            r.shape.dim[:] = s_add
            return f

        # Slicing the variable y in yolov2_activate to get t_xy
        @callback.on_generate_function_by_name('Slice')
        def slicing_t_xy(f):
            s = f.inputs[0].proto.shape.dim[:]
            s[2] = 2
            r = f.proto.slice_param
            r.stop[:] = [s[0], s[1], s[2], s[3], s[4]]
            return f

        # Arange operation in range of zero to width of input variable
        @callback.on_generate_function_by_name('Arange')
        def arange__yolov2_image_coordinate_xs(f):
            s = input_var.shape
            r = f.proto.arange_param
            r.stop = s[3]//32
            return f

        # Arange operation in range of zero to height of input variable
        @callback.on_generate_function_by_name('Arange_2')
        def arange_yolov2_image_coordinate_ys(f):
            s = input_var.shape
            r = f.proto.arange_param
            r.stop = s[2]//32
            return f

        # Slicing the variable y in yolov2_activate to get t_wh
        @callback.on_generate_function_by_name('Slice_2')
        def slicing_t_wh(f):
            s = list(f.inputs[0].proto.shape.dim[:])
            s[2] = 4
            r = f.proto.slice_param
            r.stop[:] = [s[0], s[1], s[2], s[3], s[4]]
            return f

        # Slicing the variable y in yolov2_activate to get t_o
        @callback.on_generate_function_by_name('Slice_3')
        def slicing_t_o(f):
            s = list(f.inputs[0].proto.shape.dim[:])
            s[2] = 5
            r = f.proto.slice_param
            r.stop[:] = [s[0], s[1], s[2], s[3], s[4]]
            return f

        # Slicing the variable y in yolov2_activate to get t_p
        @callback.on_generate_function_by_name('Slice_4')
        def slicing_t_p(f):
            s = list(f.inputs[0].proto.shape.dim[:])
            r = f.proto.slice_param
            r.stop[:] = [s[0], s[1], s[2], s[3], s[4]]
            return f

        # Reshape the output of Arange to get xs
        @callback.on_generate_function_by_name('Reshape_4')
        def reshape_yolov2_image_coordinate_xs(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[3] = s[0]
            return f

        # Reshape operation to get t_x
        @callback.on_generate_function_by_name('Reshape_5')
        def reshape__yolov2_image_coordinate_t_x(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[:] = [s[0], s[1], s[0]//s[0], s[2], s[3]]
            return f

        # Reshape the output of Arange_2 to get ys
        @callback.on_generate_function_by_name('Reshape_6')
        def reshape_yolov2_image_coordinate_ys(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[2] = s[0]
            return f

        # Reshape the output of Arange to get t_y
        @callback.on_generate_function_by_name('Reshape_7')
        def reshape_yolov2_image_coordinate_t_y(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[:] = [s[0], s[1], s[0]//s[0], s[2], s[3]]
            return f

        # Reshape the final variable y
        @callback.on_generate_function_by_name('Reshape_8')
        def reshape_output_variable_y(f):
            s = f.inputs[0].proto.shape.dim[:]
            r = f.proto.reshape_param
            r.shape.dim[:] = [s[0], s[1]*s[2]*s[3], s[4]]
            return f

        # Scaler division by width in function Reshape_4 in yolov2_image_coordinate
        @callback.on_generate_function_by_name('MulScalar_2')
        def mul__yolov2_image_coordinate_t_x(f):
            input_arr_shape = list(input_var.shape[2:])
            r = f.proto.mul_scalar_param
            s = f.proto.mul_scalar_param.val
            r.val = s*(nnp_input_size[1]/input_arr_shape[1])
            return f

        # Scaler division by height in function Reshape_6 in yolov2_image_coordinate
        @callback.on_generate_function_by_name('MulScalar_3')
        def mul_yolov2_image_coordinate_t_y(f):
            input_arr_shape = list(input_var.shape[2:])
            r = f.proto.mul_scalar_param
            s = f.proto.mul_scalar_param.val
            r.val = s*(nnp_input_size[0]/input_arr_shape[0])
            return f

        # Reshape biases and multiply with t_wh to rescale it.
        @callback.on_function_pass_by_name('Mul2')
        def reshape_biases(f, variables, param_scope):
            bias_param_name = f.inputs[1].proto.name
            with nn.parameter_scope('', param_scope):
                biases = nn.parameter.get_parameter(bias_param_name)
                s = list(input_var.shape)
                k = (np.array([nnp_input_size[1]//32,
                               nnp_input_size[0]//32]).reshape(1, 1, 2, 1, 1))
                m = (np.array([s[3]//32, s[2]//32]).reshape(1, 1, 2, 1, 1))
                biases = (biases.d)*k
                biases = (biases)/m
                biases = nn.Variable.from_numpy_array(biases)
                nn.parameter.set_parameter('biases', biases)

        if not training:
            callback.fix_parameters()
        batch_size = input_var.shape[0]
        net = self.nnp.get_network(
            'runtime', batch_size=batch_size, callback=callback)
        if returns_net:
            return net
        return list(net.outputs.values())[0]
