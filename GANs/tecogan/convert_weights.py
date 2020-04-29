# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
from tensorflow.python import pywrap_tensorflow
import argparse

parser = argparse.ArgumentParser(description='TecoGAN')
parser.add_argument('--pre-trained-model', default='./model/TecoGAN',
                    help='path to tensorflow pretrained model')
parser.add_argument(
    '--save-path', default='./TecoGAN_NNabla_model.h5', help='Path to save h5 file')
args = parser.parse_args()


def tf_to_nn_param_map():
    '''map from tensorflow tensor name to NNabla default parameter names 
    '''
    return {
        'Conv/weights': 'conv/W',
        'Conv/biases': 'conv/b',
        'Conv2d_transpose/weights': 'deconv/W',
        'Conv2d_transpose/biases': 'deconv/b',
    }


def rename_params(param_name):
    '''
       Rename the tensorflow tensor names to corresponding NNabla param names
    '''
    tf_to_nn_dict = tf_to_nn_param_map()
    for key in tf_to_nn_dict:
        if key in param_name:
            param_name = param_name.replace(key, tf_to_nn_dict[key])

    return param_name


def convert(ckpt_file, h5_file):
    ''' Convert the input checkpoint file to output hdf5 file
    '''
    # Get tensorflow checkpoint reader
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Loop through each tensor name from the variable to shape map
    for key in sorted(var_to_shape_map):
        # Read tensor values for each tensor name
        weight = reader.get_tensor(key)
        if(weight.ndim == 4):
            weight = np.transpose(weight, (3, 0, 1, 2))
        key = rename_params(key)

        # Create parameter with the same tensor name and shape
        params = PF.get_parameter_or_create(key, shape=weight.shape)
        params.d = weight

    # Save to a h5 file
    nn.parameter.save_parameters(h5_file)


def main():
    convert(args.pre_trained_model, args.save_path)


if __name__ == '__main__':
    main()
