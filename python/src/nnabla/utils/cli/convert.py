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

import os
import sys

import nnabla.utils.converter
from nnabla.utils.converter import get_category_info_string


def function_info_command(args):
    if args.dest is None:
        sys.stdout.write(get_category_info_string())
    else:
        with open(args.dest, 'w') as f:
            f.write(get_category_info_string())


def dump_command(args):
    if 'read_format' in args:
        if args.read_format not in nnabla.utils.converter.formats.read:
            print('Read format ({}) is not supported.'.format(args.read_format))
            return
    nnabla.utils.converter.dump_files(args, args.files)


def nnb_template_command(args):
    if 'read_format' in args:
        if args.read_format not in nnabla.utils.converter.formats.read:
            print('Read format ({}) is not supported.'.format(args.read_format))
            return
    if len(args.files) >= 2:
        output = args.files.pop()
        nnabla.utils.converter.nnb_template(args, args.files, output)


def convert_command(args):

    if 'read_format' in args:
        if args.read_format not in nnabla.utils.converter.formats.read:
            print('Read format ({}) is not supported.'.format(args.read_format))
            return

    if 'export_format' in args:
        if args.export_format not in nnabla.utils.converter.formats.export:
            print('Export format ({}) is not supported.'.format(args.export_format))
            return

    if len(args.files) >= 2:
        output = args.files.pop()
        nnabla.utils.converter.convert_files(args, args.files, output)


def add_convert_command(subparsers):
    read_formats_string = ','.join(
        nnabla.utils.converter.formats.read)

    # Function Info
    subparser = subparsers.add_parser(
        'function_info', help='Output function info.')
    subparser.add_argument('dest', nargs='?', default=None,
                           help='destination filename')
    subparser.set_defaults(func=function_info_command)

    # Dump Network.
    subparser = subparsers.add_parser(
        'dump', help='Dump network with supported format.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert.')
    # read option
    # NNP
    subparser.add_argument('-I', '--read-format', type=str, default='NNP',
                           help='[read] read format. (one of [{}])'.format(read_formats_string))
    subparser.add_argument('--nnp-expand-network', action='store_true',
                           help='[read][NNP] expand network with repeat or recurrent.')
    subparser.set_defaults(func=dump_command)

    # Generate NNB template
    subparser = subparsers.add_parser(
        'nnb_template', help='Generate NNB config file template.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert.')
    subparser.add_argument('-I', '--read-format', type=str, default='NNP',
                           help='[read] read format. (one of [{}])'.format(read_formats_string))
    subparser.add_argument('--nnp-expand-network', action='store_true',
                           help='[read][NNP] expand network with repeat or recurrent.')
    subparser.add_argument('-b', '--batch-size', type=int, default=-1,
                           help='[export] overwrite batch size.')
    subparser.add_argument('-T', '--default-variable-type', type=str, nargs=1, default=['FLOAT32'],
                           help='Default type of variable')
    subparser.set_defaults(func=nnb_template_command)

    # Conveter
    subparser = subparsers.add_parser('convert', help='File format converter.')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert.')
    # read option
    # NNP
    subparser.add_argument('-I', '--read-format', type=str, default='NNP',
                           help='[read] read format. (one of [{}])'.format(read_formats_string))
    subparser.add_argument('--nnp-expand-network', action='store_true',
                           help='[read][NNP] expand network with repeat or recurrent.')
    # export option
    export_formats_string = ','.join(
        nnabla.utils.converter.formats.export)
    subparser.add_argument('-O', '--export-format', type=str, default='NNP',
                           help='[export] export format. (one of [{}])'.format(export_formats_string))
    subparser.add_argument('-f', '--force', action='store_true',
                           help='[export] overwrite output file.')
    subparser.add_argument('-b', '--batch-size', type=int, default=-1,
                           help='[export] overwrite batch size.')
    # NNP
    subparser.add_argument('--nnp-parameter-h5', action='store_true',
                           help='[export][NNP] store parameter with h5 format')
    subparser.add_argument('--nnp-parameter-nntxt', action='store_true',
                           help='[export][NNP] store parameter into nntxt')
    subparser.add_argument('--nnp-exclude-parameter', action='store_true',
                           help='[export][NNP] output without parameter')

    # Both NNB and CSRC
    subparser.add_argument('-T', '--default-variable-type', type=str, nargs=1, default=['FLOAT32'],
                           help='Default type of variable')
    subparser.add_argument('-s', '--settings', type=str, nargs=1, default=None,
                           help='Settings in YAML format file.')

    subparser.set_defaults(func=convert_command)
