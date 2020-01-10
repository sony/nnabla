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
import nnabla.parametric_functions as pf
import numpy
import argparse

parser = argparse.ArgumentParser(description='convert vgg19')
parser.add_argument('--input_file', default="/vgg19-dcbb9e9d.pth",
                    help='pre-trained VGG19 weights from pytorch')
parser.add_argument('--output_file', default="/vgg19.h5",
                    help='path to save the converted weights')


def pytorch_to_nn_param_map(affine):
    '''map from tensor name to Nnabla default parameter names
    '''
    d1 = {'features': 'conv', 'weight': 'conv/W', 'bias': 'conv/b', '.': '/',
          "0": "1", "2": "2", "5": "3", "7": "4", "10": "5", "12": "6", "14": "7", "16": "8",
          "19": "9", "21": "10", "23": "11", "25": "12", "28": "13", "30": "14", "32": "15",
          "34": "16"}
    d2 = {"classifier": "classifier", "weight": "affine/W", "bias": "affine/b", ".": "/",
          "0": "0", "3": "3", "6": "6"}
    if not affine:
        return d1
    else:
        return d2


def rename_params(param_name, affine):
    py_to_nn_dict = pytorch_to_nn_param_map(affine)
    params = param_name.split(".")
    new_param = []
    for p in params:
        if p in py_to_nn_dict.keys():
            p = py_to_nn_dict[p]
            new_param.append(p)
    return("/".join(new_param))


def pytorch_to_nnabla(input_file, h5_file):
    read = torch.load(input_file)
    for k, v in read.items():
        if not str(k).startswith("classifier"):
            key = rename_params(str(k), affine=False)
            key = key.replace("/", "", 1)
        else:
            key = rename_params(str(k), affine=True)
        params = pf.get_parameter_or_create(key, shape=v.shape)
        params.d = v
    nn.parameter.save_parameters(h5_file)


def main():
    args = parser.parse_args()
    pytorch_to_nnabla(args.input_file, args.output_file)
    c = nn.get_parameters()
    for k, v in c.items():
        print(k)


if __name__ == "__main__":
    main()
