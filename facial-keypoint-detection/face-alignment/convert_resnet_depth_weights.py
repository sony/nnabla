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

import torch
import nnabla as nn
import nnabla.parametric_functions as PF
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='ResNetDepth weight conversion')
parser.add_argument('--pretrained-model', default='./depth-2a464da4ea.pth.tar',
                    help='path to pytorch pretrained model')
parser.add_argument('--save-path', default='./Resnet_Depth_NNabla_model.h5',
                    help='Path to save converted weight file')
args = parser.parse_args()


def pytorch_to_nn_param_map(conv):
    '''map from tensor name to Nnabla default parameter names
    '''

    d1 = OrderedDict([('fc.weight', 'fc/affine/W'), ('fc.bias', 'fc/affine/b'),
                      ('weight', 'conv/W'), ('bias', 'conv/b'), ('.', '/')])
    d2 = OrderedDict([('weight', '/bn/gamma'), ('bias', 'bn/beta'),
                      ('running_mean', '/bn/mean'), ('running_var', 'bn/var'), ('.', '/')])
    if conv:
        return d1
    else:
        return d2


def rename_params(param_name, conv):
    pytorch_to_nn_dict = pytorch_to_nn_param_map(conv)
    for k in pytorch_to_nn_dict:
        if k in param_name:
            param_name = param_name.replace(k, pytorch_to_nn_dict[k])
    return param_name


def pytorch_to_nnabla(input_file, h5_file):
    read = torch.load(input_file)
    for k, v in read['state_dict'].items():
        k = k.replace('module.', '')
        split = k.split('.')[-2]
        if split.startswith('bn') or split.startswith('1'):
            key = rename_params(k, conv=False)
            v = v.reshape((1,) + v.shape + (1, 1))
        else:
            if k == 'fc.weight':
                v = v.T
            key = rename_params(k, conv=True)
        params = PF.get_parameter_or_create(key, shape=v.shape)
        params.d = v.cpu().numpy()
    nn.parameter.save_parameters(h5_file)


def main():
    pytorch_to_nnabla(args.pretrained_model, args.save_path)


if __name__ == "__main__":
    main()
