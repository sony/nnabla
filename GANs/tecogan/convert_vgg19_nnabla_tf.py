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

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import argparse

parser = argparse.ArgumentParser(description='convert vgg19')
parser.add_argument('--input_file', default="/home/ubuntu/Desktop/tecogan/vgg19.h5",
                    help='pre-trained VGG19 nnabla weights')
parser.add_argument('--output_file', default="./vgg19_tf.h5",
                    help='path to save the converted weights')


def nnabla_to_tf_nnabla(input_file, h5_file):
    with nn.parameter_scope("vgg19"):
        nn.load_parameters(input_file)
        read = nn.get_parameters()
    for k, v in read.items():
        if v.ndim == 4:
            v = F.transpose(v, (0, 2, 3, 1))
        if v.ndim == 2:
            v = F.transpose(v, (1, 0))
        with nn.parameter_scope("vgg19_tf"):
            params = PF.get_parameter_or_create(k, shape=v.shape)
    with nn.parameter_scope("vgg19_tf"):
        nn.parameter.save_parameters(h5_file)


def main():
    args = parser.parse_args()
    nnabla_to_tf_nnabla(args.input_file, args.output_file)
    with nn.parameter_scope("vgg19_tf"):
        c = nn.get_parameters()
    for k, v in c.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
