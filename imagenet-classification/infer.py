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


def normalize_uint8_image(image):
    from normalize_config import _pixel_mean, _pixel_std
    mean = np.asarray(_pixel_mean, dtype=np.float32)
    std = np.asarray(_pixel_std, dtype=np.float32)
    image = (image - mean) / std
    return image


def crop_center_image(image, crop_shape):
    H, W = image.shape[:2]
    h, w = crop_shape
    hh, ww = (H - h) // 2, (W - w) // 2
    image = image[hh:hh+h, ww:ww+w]
    return image


def read_image_with_preprocess(path, channel_last=False,
                               channels=3):
    assert channels in (3, 4)
    from nnabla.utils.image_utils import imread
    H, W = 256, 256
    h, w = 224, 224
    image = imread(path, num_channels=3, size=(W, H))
    image = crop_center_image(image, (h, w))
    image = normalize_uint8_image(image)
    if channels == 4:
        shape = list(image.shape)
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
    parser = argparse.ArgumentParser(
        description='Inference.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension module. 'cudnn' is highly.recommended.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration.')
    parser.add_argument("--num-layers", "-L", type=int,
                        choices=[18, 34, 50, 101, 152], default=50,
                        help='Number of layers of ResNet.')
    parser.add_argument("--shortcut-type", "-S", type=str,
                        choices=['b', 'c', ''], default='b',
                        help='Skip connection type. See `resnet_imagenet()` in model_resent.py for description.')
    parser.add_argument("input_image", help='Path to an input image.')
    parser.add_argument("weights", help='Path to a trained parameter h5 file.')
    parser.add_argument(
        '--labels', help='Path to a label name file which contain label names as csv compatible with `label_words.csv`.', default='./label_words.csv')
    return parser.parse_args()


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
        args.input_image, channel_last=channel_last,
        channels=channels)
    img = nn.NdArray.from_numpy_array(image)

    # Perform inference
    import model_resnet_nhwc
    num_classes = 1000
    pred, _ = model_resnet_nhwc.resnet_imagenet(
        img, num_classes, args.num_layers, args.shortcut_type, test=True, tiny=False, channel_last=channel_last)
    prob = F.softmax(pred)
    top5_index = F.sort(prob, reverse=True, only_index=True)[:, :5]

    # Get and print result
    labels = read_labels(args.labels)
    logger.info(f'Top-5 prediction:')
    for i in top5_index.data[0]:
        logger.info(f'* {labels[int(i)]}: {prob.data[0, int(i)] * 100:.2f}')


if __name__ == '__main__':
    main()
