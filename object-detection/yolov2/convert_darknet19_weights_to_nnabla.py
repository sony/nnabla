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


import darknet_parser as parser
import nnabla as nn
import darknet19
import numpy as np


def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='darknet19.weights')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    if args.output is None:
        args.output = args.input.replace('.weights', '.h5')
    return args


def main():
    args = get_args()

    # Defining network first
    x = nn.Variable((1, 3, 224, 224))
    y = darknet19.darknet19_classification(x / 255, test=True)

    # Get NNabla parameters
    params = nn.get_parameters(grad_only=False)

    # Parse Darknet weights and store them into NNabla params
    dn_weights = parser.load_weights_raw(args.input)
    cursor = 0
    for i in range(1, 19):  # 1 to 18
        cursor = parser.load_convolutional_and_get_next_cursor(
            dn_weights, cursor, params, 'c{}'.format(i))
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'c19', no_bn=True)

    nn.save_parameters(args.output)


if __name__ == '__main__':
    main()
