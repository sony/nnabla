# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

import torch
import nnabla as nn
import nnabla.parametric_functions as PF
import numpy
import argparse

parser = argparse.ArgumentParser(description='esrgan')
parser.add_argument('--pretrained_model', default='./RRDB_ESRGAN_x4.pth',
                    help='path to pytorch pretrained model')
parser.add_argument('--save_path', default='./ESRGAN_NNabla_model.h5',
                    help='Path to save h5 file')
args = parser.parse_args()


def pytorch_to_nn_param_map():
    '''map from tensor name to Nnabla default parameter names
    '''
    return {
        'weight': 'conv/W',
        'bias': 'conv/b',
        '.': '/'
    }


def rename_params(param_name):
    pytorch_to_nn_dict = pytorch_to_nn_param_map()
    for k in pytorch_to_nn_dict:
        if k in param_name:
            param_name = param_name.replace(k, pytorch_to_nn_dict[k])
    return param_name


def pytorch_to_nnabla(input_file, h5_file):
    read = torch.load(input_file)
    for k, v in read.items():
        key = rename_params(k)
        params = PF.get_parameter_or_create(key, shape=v.shape)
        params.d = v
    nn.parameter.save_parameters(h5_file)


def main():
    pytorch_to_nnabla(args.pretrained_model, args.save_path)


if __name__ == "__main__":
    main()
