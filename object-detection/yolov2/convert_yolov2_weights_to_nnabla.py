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

import numpy as np

import darknet_parser as parser
import darknet19
import yolov2


def get_args():
    ''' Parse arguments
    '''
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='yolov2.weights')
    p.add_argument('--output', default=None)
    p.add_argument('--anchors', type=int, default=5)
    p.add_argument('--classes', type=int, default=80)
    p.add_argument('--width', type=int, default=608)
    p.add_argument('--header', type=int, default=4)
    args = p.parse_args()
    if args.output is None:
        args.output = args.input.replace('.weights', '.h5')
    return args


def main():
    '''Main.
    '''

    # Parse arguments
    args = get_args()

    # Create YOLOv2 detection newtork
    x = nn.Variable((1, 3, args.width, args.width))
    y = yolov2.yolov2(x, args.anchors, args.classes, test=True)
    params = nn.get_parameters(grad_only=False)

    # Parse network parameters
    dn_weights = np.fromfile(args.input, dtype=np.float32)[args.header:]
    cursor = 0
    for i in range(1, 19):  # 1 to 18
        cursor = parser.load_convolutional_and_get_next_cursor(
            dn_weights, cursor, params, 'c{}'.format(i))
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'c18_19')
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'c18_20')
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'c13_14')
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'c21')
    cursor = parser.load_convolutional_and_get_next_cursor(
        dn_weights, cursor, params, 'detection', no_bn=True)
    assert cursor == dn_weights.size

    # Save to a h5 file
    nn.save_parameters(args.output)


if __name__ == '__main__':
    main()
