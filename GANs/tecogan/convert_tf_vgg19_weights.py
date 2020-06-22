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

parser = argparse.ArgumentParser(
    description='Convert TF VGG19 weights to NNabla format')
parser.add_argument('--pre_trained_model', default='./vgg_19.ckpt',
                    help='path to tensorflow pretrained model')
parser.add_argument('--save-path', default='./VGG19_NNabla_model.h5',
                    help='Path to save h5 file')
args = parser.parse_args()


def tf_to_nn_param_map(affine):
    """ 
    Map from tensorflow default param names to NNabla default param names 
    """
    d1 = {'weights': 'conv/W', 'biases': 'conv/b',
          "conv1_1": "conv1", "conv1_2": "conv2", "conv2_1": "conv3", "conv2_2": "conv4", "conv3_1": "conv5",
          "conv3_2": "conv6", "conv3_3": "conv7", "conv3_4": "conv8", "conv4_1": "conv9", "conv4_2": "conv10",
          "conv4_3": "conv11", "conv4_4": "conv12", "conv5_1": "conv13", "conv5_2": "conv14", "conv5_3": "conv15",
          "conv5_4": "conv16", "vgg19": ""}
    d2 = {"fc6": "classifier/0", "weights": "affine/W", "biases": "affine/b", "fc7": "classifier/3",
          "fc8": "classifier/6", "vgg19": ""}
    if not affine:
        return d1
    else:
        return d2


def rename_params(param_name, affine):
    """ 
    Rename the tensorflow param names to corresponding NNabla param names
    """
    py_to_nn_dict = tf_to_nn_param_map(affine)
    params = param_name.split("/")
    new_param = []
    for p in params:
        if p in py_to_nn_dict.keys():
            p = py_to_nn_dict[p]
            new_param.append(p)
    return("/".join(new_param))


def convert_ckpt_to_h5(input_file, h5_file):
    """ 
    Convert the input checkpoint file to output hdf5 file
    """
    # Get tensorflow checkpoint reader
    reader = pywrap_tensorflow.NewCheckpointReader(input_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Loop through each tensor name from the variable to shape map
    for key in var_to_shape_map:
        # Read tensor values for each tensor name
        weight = reader.get_tensor(key)
        if not str(key).startswith("vgg_19/mean_rgb") and not str(key).startswith("global_step"):
            s = key.split('/')
            if str(s[-2]).startswith("fc"):
                k = ('/'.join(s))
                key = rename_params(str(k), affine=True)
            else:
                s.remove(s[1])
                k = ('/'.join(s))
                key = rename_params(str(k), affine=False)
            if(weight.ndim == 4):
                # transpose TF weight to NNabla weight format
                weight = np.transpose(weight, (3, 0, 1, 2))

                # Create parameter with the same tensor name and shape
            params = PF.get_parameter_or_create(key, shape=weight.shape)
            params.d = weight

        # Save to a h5 file
    nn.parameter.save_parameters(h5_file)


def main():
    convert_ckpt_to_h5(args.pre_trained_model, args.save_path)


if __name__ == '__main__':
    main()
