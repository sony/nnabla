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

import argparse


class Yolov2Option(object):
    def __init__(self, *options):
        self.options = options

    def add_arguments(self, parser):
        for o in self.options:
            o.add_arguments(parser)
        self.add_arguments_impl(parser)

    def post_process(self, parser):
        for o in self.options:
            o.post_process(parser)
        self.post_process_impl(parser)

    def add_arguments_impl(self, parser):
        pass

    def post_process_impl(self, args):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        args = parser.parse_args()
        self.post_process(args)
        return args


def get_anchors_by_name_or_parse(anchors):
    if anchors == 'voc':
        anchors = [1.3221, 1.73145, 3.19275, 4.00944,
                   5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
    elif anchors == 'coco':
        anchors = [0.57273, 0.677385, 1.87446, 2.06253,
                   3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    else:
        anchors = [float(x.strip()) for x in anchors.split(",")]
    return anchors


class Yolov2OptionCommon(Yolov2Option):
    def add_arguments_impl(self, parser):
        parser.add_argument('--num-classes', type=int,
                            help='Number of classes', default=20)
        parser.add_argument('-a', '--anchors', type=str,
                            help='Anchors {"voc", "coco", or (w0, h0, w1, h1, ..., wN, hN)}.',
                            default="voc")
        parser.add_argument('-g', '--gpus', type=str,
                            help='GPU IDs to be used', default="0")
        parser.add_argument('--disable-cv2', action='store_true')
        parser.add_argument('--fine-tune', action='store_true', default=False,
                            help="Whether to fine tune model or not; False by default")

    def post_process_impl(self, args):
        args.anchors = get_anchors_by_name_or_parse(args.anchors)
        args.num_anchors = len(args.anchors)//2
        args.use_cuda = args.gpus != "-1"


class Yolov2OptionLog(Yolov2Option):
    def add_arguments_impl(self, parser):
        parser.add_argument('-o', '--output', type=str,
                            help='Weight output directory', default="backup")
        parser.add_argument('-w', '--weight', type=str,
                            help='Initial weight file', required=True)


class Yolov2OptionInfer(Yolov2Option):
    def add_arguments_impl(self, parser):
        parser.add_argument('--width', type=int,
                            help='Input image width', default=416)
        parser.add_argument('--height', type=int,
                            help='Input image height', default=416)
        parser.add_argument('-n', '--names', type=str,
                            help='Class name list', default="data/voc.names")


class Yolov2OptionTraining(Yolov2Option):
    def __init__(self, *options):
        super(Yolov2OptionTraining, self).__init__(
            Yolov2OptionCommon(),
            Yolov2OptionLog(),
            *options)

    def add_arguments_impl(self, parser):
        parser.add_argument('--size-aug', type=str,
                            help='The Python expression of the list of image sizes for augmentation.',
                            default='tuple(range(320, 608 + 1, 32))')
        parser.add_argument('-t', '--train', type=str,
                            help='Training dataset list', default="dataset/train.txt")
        parser.add_argument('-b', '--batch-size', type=int,
                            help='Batch size', default=8)
        parser.add_argument('--accum-times', type=int,
                            help='Number of times of batch accumulation. e.g. when batch_size == 64 and accum_times == 2, the gradient is calculated by accumulating a minibatch of size 32 twice.', default=8)
        parser.add_argument('--max-batches', type=int,
                            help='Maximum number of batches', default=80200)
        parser.add_argument('--learning-rate', type=float,
                            help='Learning rate', default=0.001)
        parser.add_argument('--momentum', type=float,
                            help='Momentum', default=0.9)
        parser.add_argument('--decay', type=float,
                            help='Weight decay coefficient', default=0.0005)
        parser.add_argument('--steps', type=str,
                            help='Steps', default="40000,60000")
        parser.add_argument('--scales', type=str,
                            help='Scales', default=".1, .1")
        parser.add_argument('--burn-in', type=int,
                            help='Burn-in learning rate.', default=1000)
        parser.add_argument('--burn-in-power', type=float,
                            help='Burn-in power.', default=4.0)
        parser.add_argument('--save-interval', type=int,
                            help='Interval of epochs to save weight files', default=10)
        # Data augmentation parameters
        parser.add_argument('--jitter', type=float,
                            help='Data augmentation: jitter', default=0.2)
        parser.add_argument('--hue', type=float,
                            help='Data augmentation: hue', default=0.1)
        parser.add_argument('--saturation', type=float,
                            help='Data augmentation: saturation', default=1.5)
        parser.add_argument('--exposure', type=float,
                            help='Data augmentation: exposure', default=1.5)
        parser.add_argument('--on-memory-data',
                            action='store_true', default=False)
        parser.add_argument('--object-scale', type=float,
                            help='Object Scale', default=5.0)
        parser.add_argument('--noobject-scale',
                            type=float, help='No-object Scale', default=1.0)
        parser.add_argument('--class-scale', type=float,
                            help='Class Scale', default=1.0)
        parser.add_argument('--coord-scale',
                            type=float, help='Coord Scale', default=1.0)
        parser.add_argument('--thresh', type=float, default=0.6)

    def post_process_impl(self, args):
        args.size_aug = eval(args.size_aug)
        args.steps = [int(x.strip()) for x in args.steps.split(",")]
        args.scales = [float(x.strip()) for x in args.scales.split(",")]


class Yolov2OptionValid(Yolov2Option):
    def __init__(self, *options):
        super(Yolov2OptionValid, self).__init__(
            Yolov2OptionCommon(),
            Yolov2OptionInfer(),
            Yolov2OptionLog(), *options)

    def add_arguments_impl(self, parser):
        parser.add_argument('-v', '--valid', type=str,
                            help='Validation dataset list', default="dataset/2007_test.txt")
        parser.add_argument('--conf-thresh', type=float,
                            help='Confidence threshhold', default=0.005)
        parser.add_argument('--nms-thresh', type=float,
                            help='IOU threshhold for non-maximum suppression', default=0.45)
        parser.add_argument('--valid-batchsize',
                            type=int, help='Batch size', default=2)
