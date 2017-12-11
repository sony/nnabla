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


def function_info_command(args):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'converter', 'function_info.json')) as f:
        info = f.read()
    if args.dest is None:
        sys.stdout.write(info)
    else:
        with open(args.dest, 'w') as f:
            f.write(info)


def convert_command(args):
    print(args.files)
    if len(args.files) >= 2:
        output = args.files.pop()
        result = True
        for ifile in args.files:
            if not os.path.isfile(ifile):
                print('Input file ({}) does not exist.'.format(ifile))
                result = False
        if result:
            if os.path.splitext(output)[1] in nnabla.utils.converter.supported_info.extensions_for_export:
                if os.path.isfile(output):
                    if args.force:
                        nnabla.utils.converter.convert_files(
                            args, args.files, output)
                    else:
                        print('Output file ({}) already exists.'.format(output))
                        print(' Remove it or specify `-f` option.')
                        result = False
                else:
                    if os.path.isdir(os.path.dirname(os.path.abspath(output))):
                        nnabla.utils.converter.convert_files(
                            args, args.files, output)
                    else:
                        print('Output directory ({}) does not exist.'.format(
                            os.path.dirname(output)))
                        print(' Make output directory beforre convert.')
                        result = False

            elif os.path.isdir(output):
                nnabla.utils.converter.convert_files(args, args.files, output)
            else:
                print('Output ({}) is invalid.'.format(output))
                print(' Please specify output file or existing directory.')
                result = False

    return result


def add_convert_command(subparsers):
    # Function Info
    subparser = subparsers.add_parser('function_info')
    subparser.add_argument('dest', nargs='?', default=None,
                           help='destination filename')
    subparser.set_defaults(func=function_info_command)

    # Conveter
    subparser = subparsers.add_parser('convert')
    subparser.add_argument('files', metavar='FILE', type=str, nargs='+',
                           help='File or directory name(s) to convert.')
    # general option
    # read option
    # NNP
    read_formats_string = ','.join(
        [x[1:] for x in nnabla.utils.converter.supported_info.extensions_for_read])
    subparser.add_argument('-I', '--read-format', type=str, default=None,
                           help='[read] read format. (one of [{}])'.format(read_formats_string))
    subparser.add_argument('--nnp-expand-network', action='store_true',
                           help='[read][NNP] expand network with repeat or recurrent.')
    # export option
    export_formats_string = ','.join(
        [x[1:] for x in nnabla.utils.converter.supported_info.extensions_for_export])
    subparser.add_argument('-O', '--export-format', type=str, default=None,
                           help='[export] export format. (one of [{}])'.format(export_formats_string))
    subparser.add_argument('-f', '--force', action='store_true',
                           help='[export] overwrite output file.')
    # NNP
    subparser.add_argument('--nnp-parameter-h5', action='store_true',
                           help='[export][NNP] store parameter with h5 format')
    subparser.add_argument('--nnp-parameter-nntxt', action='store_true',
                           help='[export][NNP] store parameter into nntxt')
    subparser.add_argument('--nnp-exclude-parameter', action='store_true',
                           help='[export][NNP] output without parameter')
    subparser.set_defaults(func=convert_command)
