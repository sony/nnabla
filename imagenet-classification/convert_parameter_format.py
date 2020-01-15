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

'''
Convert parameter format such as data memory layout NHWC <--> NCHW etc.
'''

import nnabla as nn
import numpy as np
from nnabla import logger


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Inference.')
    parser.add_argument("input", help='Path to an input h5 parameter file.')
    parser.add_argument(
        "output", help='Path to an output h5 paramter file which will be created.')
    parser.add_argument('--memory-layout', '-m',
                        help='Convert weight to "NHWC" or "NCHW".', default=None)
    parser.add_argument('--force-3-channels', '-3', action='store_true', default=False,
                        help='Remove padded 4-th channel in first layer convolution.')
    args = parser.parse_args()
    if args.memory_layout is not None:
        args.memory_layout = args.memory_layout
        assert args.memory_layout in (
            'nhwc', 'nchw'), 'Memory layout target must be either "nhwc" or "nchw": {args.memory_layout}'
    return args


def get_memory_layout(params):
    for key in params.keys():
        if key.endswith('bn/beta'):
            break
    if params[key].shape[1] == 1:
        return 'nhwc'
    return 'nchw'


def convert_memory_layout(params, layout):
    for key in params.keys():
        if key.endswith('conv/b'):
            continue
        param = params[key]
        array = param.d
        if layout == 'nhwc':
            array = np.transpose(array, (0, 2, 3, 1))
        else:
            # 'nhwc' --> 'nchw'
            array = np.transpose(array, (0, 3, 1, 2))
        new_param = nn.Variable.from_numpy_array(array)
        new_param.need_grad = param.need_grad
        params[key] = new_param


def force_3_channels(params, layout):
    key = 'conv1/conv/W'
    param = params[key]
    channels = param.shape[3] if layout == 'nhwc' else param.shape[1]
    if channels == 3:
        return False
    array = param.d
    if layout == 'nhwc':
        array = array[..., :3]
    else:
        array = array[:, :3]
    new_param = nn.Variable.from_numpy_array(array)
    new_param.need_grad = param.need_grad
    params[key] = new_param
    return True


def main():
    args = get_args()

    nn.load_parameters(args.input)
    params = nn.get_parameters(grad_only=False)

    processed = False

    # Convert memory layout
    layout = get_memory_layout(params)
    if args.memory_layout is None:
        pass
    elif args.memory_layout != layout:
        logger.info(f'Converting memory layout to {args.memory_layout}.')
        convert_memory_layout(params, args.memory_layout)
        processed |= True
    else:
        logger.info('No need to convert memory layout.')

    if args.force_3_channels:
        ret = force_3_channels(params, args.memory_layout)
        if ret:
            logger.info('Converted first conv to 3-channel input.')
        processed |= ret

    if not processed:
        logger.info(
            'No change has been made for the input. Not saving a new parameter file.')
        return
    logger.info(f'Save a new parameter file at {args.output}')
    for key, param in params.items():
        nn.parameter.set_parameter(key, param)
    nn.save_parameters(args.output)


if __name__ == '__main__':
    main()
