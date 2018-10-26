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


import yolov2

import nnabla as nn
import nnabla.functions as F

import time
import numpy as np
from nnabla.utils.image_utils import imread, imresize, imsave


from arg_utils import get_anchors_by_name_or_parse


def get_args():
    import argparse
    from os.path import dirname, basename, join
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=608)
    p.add_argument('--height', type=int, default=608)
    p.add_argument('--weights', type=str, default='yolov2.h5')
    p.add_argument('--classes', type=int, default=80)
    p.add_argument('--thresh', type=float, default=.5)
    p.add_argument('--nms', type=float, default=.45)
    p.add_argument('--nms-per-class', type=bool, default=True)
    p.add_argument(
        '--anchors', type=str,
        default='coco')
    p.add_argument('--nnp', type=str, default='yolov2.nnp')
    args = p.parse_args()
    assert args.width % 32 == 0
    assert args.height % 32 == 0
    args.anchors = get_anchors_by_name_or_parse(args.anchors)
    args.num_anchors = int(len(args.anchors) // 2)
    args.anchors = np.array(args.anchors).reshape(-1, 2)
    return args


def main():
    args = get_args()

    # Load parameter
    _ = nn.load_parameters(args.weights)

    # Build a YOLO v2 network
    x = nn.Variable((1, 3, args.height, args.width))
    y = yolov2.yolov2(x / 255.0, args.num_anchors, args.classes, test=True)
    y = yolov2.yolov2_activate(y, args.num_anchors, args.anchors)
    y = F.nms_detection2d(y, args.thresh, args.nms, args.nms_per_class)

    # Save NNP file (used in C++ inference later.).
    runtime_contents = {
        'networks': [
            {'name': 'runtime',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'data': ['x'],
             'output': ['y']}]}
    import nnabla.utils.save
    nnabla.utils.save.save(args.nnp, runtime_contents)


if __name__ == '__main__':
    main()
