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

import nnabla as nn
import nnabla.functions as F
from nnabla import logger

import numpy as np


def normalize_uint8_image(image, config='default'):
    from normalize_config import get_normalize_config
    mean, std = get_normalize_config(config)
    image = image - mean
    if std is not None:
        image /= std
    return image


def crop_center_image(image, crop_shape):
    H, W = image.shape[:2]
    h, w = crop_shape
    hh, ww = (H - h) // 2, (W - w) // 2
    image = image[hh:hh+h, ww:ww+w]
    return image


def read_image_with_preprocess(path, norm_config, channel_last=False,
                               channels=3):
    assert channels in (3, 4)
    from nnabla.utils.image_utils import imread
    H, W = 256, 256
    h, w = 224, 224
    image = imread(path, num_channels=3, size=(W, H))
    image = crop_center_image(image, (h, w))
    image = normalize_uint8_image(image, norm_config)
    if channels == 4:
        image = np.pad(image, ((0, 0), (0, 0), (0, 1)),
                       mode='constant', constant_values=0)
    if not channel_last:
        image = np.transpose(image, (2, 0, 1))
    return image[None]  # Add batch dimension


def read_labels(path):
    import csv
    labels = [row[1] for row in csv.reader(open(path))]
    return labels


def get_args():
    import argparse
    import args as A
    parser = argparse.ArgumentParser(
        description='Inference.')
    A.add_runtime_args(parser)
    A.add_arch_args(parser)
    parser.add_argument("input_image", help='Path to an input image.')
    parser.add_argument("weights", help='Path to a trained parameter h5 file.')
    parser.add_argument(
        '--labels', help='Path to a label name file which contain label names as csv compatible with `label_words.csv`.', default='./label_words.csv')
    parser.add_argument(
        '--norm-config', '-n', type=A.lower_str, default='default',
        help='Specify how to normalize an image as preprocessing.')
    args = parser.parse_args()

    # See available archs
    A.check_arch_or_die(args.arch)
    return args


def load_parameters_and_config(path):
    '''
    Load paramters and deduce the configuration
    of memory layout and input channels

    Returns: (channel_last, input_channels)

    '''
    nn.load_parameters(path)
    try:
        conv1 = nn.parameter.get_parameter('conv1/conv/W')
    except:
        raise ValueError(
            'conv1/conv/W is not found. This parameter configuration deduction works for resnet only.')
    shape = conv1.shape
    assert shape[1] == 7 or shape[3] == 7, 'This deduction process assumes that the first convolution has 7x7 filter.'
    channel_last = False
    channels = shape[1]
    if shape[1] == 7:
        channel_last = True
        channels = shape[3]
    assert channels in (3, 4), f'channels must be either 3 or 4: {channels}.'
    return channel_last, channels


def main():
    args = get_args()

    # Setup
    from nnabla.ext_utils import get_extension_context
    if args.context is None:
        extension_module = "cudnn"  # TODO: Hard coded!!!
    else:
        extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Load parameters
    channel_last, channels = load_parameters_and_config(args.weights)
    logger.info('parameter configuration is deduced as:')
    logger.info(f'* channel_last={channel_last}')
    logger.info(f'* channels={channels}')

    # Read image
    image = read_image_with_preprocess(
        args.input_image, args.norm_config, channel_last=channel_last,
        channels=channels)
    img = nn.NdArray.from_numpy_array(image)

    # Perform inference
    from models import build_network
    num_classes = 1000
    pred, _ = build_network(img, num_classes, args.arch,
                            test=True, channel_last=channel_last)
    prob = F.softmax(pred)
    top5_index = F.sort(prob, reverse=True, only_index=True)[:, :5]

    # Get and print result
    labels = read_labels(args.labels)
    logger.info(f'Top-5 prediction:')
    for i in top5_index.data[0]:
        logger.info(f'* {int(i)} {labels[int(i)]}: {prob.data[0, int(i)] * 100:.2f}')


if __name__ == '__main__':
    main()
