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
import nnabla.logger as logger
import nnabla.functions as F
import model_resnet

import os
from collections import namedtuple

from nnabla.utils.image_utils import imresize, imread

import argparse
import csv

LABEL_AND_NAME = "label_words.csv"


def get_model(args, num_classes):
    """
    Create computation graph and variables.
    """
    data_size = 224
    image = nn.Variable([1, 3, data_size, data_size])
    pimage = image_preprocess(image)
    pred, hidden = model_resnet.resnet_imagenet(
        pimage, num_classes, args.num_layers, args.shortcut_type, test=True, tiny=False)
    Model = namedtuple('Model', ['image', 'pred', 'hidden'])
    return Model(image, pred, hidden)


def image_preprocess(image):
    image = image / 255.0
    image = image - 0.5
    return image


def resize_and_crop_center(im):
    # resize
    width = 256
    height = 256
    h = im.shape[0]
    w = im.shape[1]
    # trimming mode
    if float(h) / w > float(height) / width:
        target_h = int(float(w) / width * height)
        im = im[(h - target_h) // 2:h -
                (h - target_h) // 2, ::]
    else:
        target_w = int(float(h) / height * width)
        im = im[::, (w - target_w) // 2:w -
                (w - target_w) // 2]
    # crop
    im = imresize(im, (width, height))
    hc = im.shape[0] // 2
    wc = im.shape[1] // 2
    r = 224 // 2
    hs = hc - r
    he = hc + r
    ws = wc - r
    we = wc + r
    x = np.array(im[hs:he, ws:we], dtype=np.uint8).transpose((2, 0, 1))
    return x


def print_result(labels, values):
    with open(LABEL_AND_NAME, 'r') as f:
        reader = csv.reader(f)
        print("      label: \"words\" [predicted value]")
        for i, (j, k) in enumerate(zip(labels[0][:5], values[0][:5])):
            f.seek(0)
            for l in reader:
                if j == int(l[0]):
                    print("Top-%d:" % (i + 1), format(j, '3'),
                          ':', '"'+l[1]+'"', "[%e]" % k)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", "-i", type=str,
                        default=None, help='input image file')
    parser.add_argument("--weight-file", "-w", type=str,
                        default=None, help='input parameter file')
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension module. 'cudnn' is highly.recommended.")
    parser.add_argument("--num-layers", "-L", type=int,
                        choices=[18, 34, 50, 101, 152], default=34,
                        help='Number of layers of ResNet.')
    parser.add_argument("--shortcut-type", "-S", type=str,
                        choices=['b', 'c', ''], default='b',
                        help='Skip connection type. See `resnet_imagenet()` in model_resent.py for description.')
    return parser.parse_args()


def infer():
    """
    Main script.
    """

    # get args.
    args = get_args()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config
        )
    nn.set_default_context(ctx)
    nn.clear_parameters()  # To infer.

    # Get data from args.
    im = imread(args.input_file, num_channels=3)
    vdata = resize_and_crop_center(im)

    # Get a model.
    num_classes = 1000  # The number of class.
    v_model = get_model(args, num_classes)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward

    # Get parameters from parameter file.
    nn.load_parameters(args.weight_file)

    # Perfome inference.
    v_model.image.d = vdata
    v_model.image.data.cast(np.uint8, ctx)
    v_model.pred.forward(clear_buffer=True)
    values, labels = F.sort(-v_model.pred.data, with_index=True)
    ratios = F.softmax(-values)
    print_result(labels.data, ratios.data)


if __name__ == '__main__':
    infer()
