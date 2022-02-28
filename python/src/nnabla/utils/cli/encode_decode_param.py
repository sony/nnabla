# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import print_function

import os
import urllib.parse

import nnabla as nn
import numpy as np
from nnabla import logger
from nnabla.parameter import get_parameters, set_parameter, save_parameters, load_parameters


def save_param_in_txt(x, filepath):
    with open(filepath, 'w') as output:
        print(x.shape, file=output)
        np.array(x).tofile(output, sep="\n")


def load_param_in_txt(name, filepath):
    with open(filepath, 'r') as input:
        ds = input.read()
    ds = ds.split("\n")
    shape = eval(ds.pop(0))
    variable = nn.Variable(shape, need_grad=True)
    variable.d = np.fromstring("\n".join(ds),
                               dtype=np.float32,
                               sep="\n").reshape(shape)
    set_parameter(name, variable)


# subcommands
# ===========
def decode_param_command(args, **kwargs):
    os.makedirs(args.outdir, exist_ok=True)

    # Load parameter
    logger.log(99, 'Loading parameters...')
    load_parameters(args.param)

    # Save Parameters
    params = get_parameters(grad_only=False)
    for key, variable in params.items():
        logger.log(99, key)
        file_path = os.path.join(args.outdir,
                                 urllib.parse.quote(key, safe='/ ').replace('/', '~') + '.txt')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_param_in_txt(variable.d, file_path)

    logger.log(99, 'Decode Parameter Completed.')
    return True


def encode_param_command(args, **kwargs):
    # Load Parameters
    in_files = [f for f in os.listdir(
        args.indir) if os.path.isfile(os.path.join(args.indir, f))]
    logger.log(99, 'Loading parameters...')
    for file_path in in_files:
        key = urllib.parse.unquote(
            os.path.splitext(file_path)[0].replace('~', '/'))
        logger.log(99, key)
        load_param_in_txt(key, os.path.join(args.indir, file_path))
    # Save parameter
    logger.log(99, 'Saving parameters...')
    save_parameters(args.param)

    logger.log(99, 'Encode Parameter Completed.')
    return True


def add_decode_param_command(subparsers):
    # Decode param
    subparser = subparsers.add_parser(
        'decode_param', help='Decode parameter to plain text.')
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=decode_param_command)


def add_encode_param_command(subparsers):
    # Encode param
    subparser = subparsers.add_parser(
        'encode_param', help='Encode plain text to parameter format.')
    subparser.add_argument(
        '-i', '--indir', help='input directory', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.set_defaults(func=encode_param_command)
