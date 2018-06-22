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

import collections
import os
import sys

from .nnabla import NnpReader, NnpExporter
from .nnablart import NnbExporter, CsrcExporter
from .onnx import OnnxReader, OnnxExporter


def _read_nnp(args, ifiles):
    if len(ifiles) == 1 and os.path.splitext(ifiles[0])[1] == '.nnp':
        args.read_format = 'NNP'
    if len(ifiles) == 1 and os.path.splitext(ifiles[0])[1] == '.onnx':
        args.read_format = 'ONNX'
    if args.read_format == 'NNP':
        # Input file that has unsuported extension store into output nnp archive or directory.
        return NnpReader(*ifiles, expand_network=args.nnp_expand_network).read()
    elif args.read_format == 'ONNX':
        return OnnxReader(*ifiles).read()
    return None


def _export_from_nnp(args, nnp, output):
    output_ext = os.path.splitext(output)[1].lower()
    if (os.path.isdir(output) and args.export_format == 'NNP') or output_ext == '.nnp':
        parameter_type = 'protobuf'
        if args.nnp_parameter_nntxt:
            parameter_type = 'included'
        elif args.nnp_parameter_h5:
            parameter_type = 'h5'
        if args.nnp_exclude_parameter:
            parameter_type = 'none'
        NnpExporter(nnp, args.batch_size, parameter_type).export(output)

    elif output_ext == '.nnb':
        NnbExporter(nnp, args.batch_size).export(
            output, None, args.settings, args.default_variable_type)

    elif os.path.isdir(output) and args.export_format == 'CSRC':
        CsrcExporter(nnp, args.batch_size).export(output)

    elif output_ext == '.onnx':
        OnnxExporter(nnp, args.batch_size).export(output)
    else:
        print('Output file ({}) is not supported or output directory does not exist.'.format(
            output_ext))
        return False
    return True


def convert_files(args, ifiles, output):
    nnp = _read_nnp(args, ifiles)
    if nnp is not None:
        return _export_from_nnp(args, nnp, output)
    else:
        print('Read from [{}] failed.'.format(ifiles))
        return False


def _generate_nnb_template(args, nnp, output):
    NnbExporter(nnp, args.batch_size).export(
        None, output, None, args.default_variable_type)
    return True


def nnb_template(args, ifiles, output):
    nnp = _read_nnp(args, ifiles)
    if nnp is not None:
        return _generate_nnb_template(args, nnp, output)
    else:
        print('Read from [{}] failed.'.format(ifiles))
        return False


def _dump_protobuf(proto, prefix, depth):
    if depth >= 0 and len(prefix) >= depth:
        print('{} ...'.format(':'.join(prefix)))
        return
    for desc, field in proto.ListFields():
        if isinstance(field, (int, float, complex, str)):
            print('{}:{}: {}'.format(':'.join(prefix), desc.name, field))
        elif isinstance(field, collections.Iterable):
            print('{} has {} {}(s).'.format(
                ':'.join(prefix), len(field), desc.name))
            for n, f in enumerate(field[:10]):
                if isinstance(f, (int, float, complex, str)):
                    print('{}:{}[{}]: {}'.format(
                        ':'.join(prefix), desc.name, n, f))
                else:
                    if depth < 0 or depth > len(prefix)+1:
                        _dump_protobuf(
                            f, prefix + ['{}[{}]'.format(desc.name, n)], depth)
        else:
            _dump_protobuf(field, prefix + [desc.name], depth)


def _dump_nnp(args, nnp):
    _dump_protobuf(nnp.protobuf, [args.read_format], -1)
    return True


def dump_files(args, ifiles):
    nnp = _read_nnp(args, ifiles)
    if nnp is not None:
        return _dump_nnp(args, nnp)
    else:
        print('Read from [{}] failed.'.format(ifiles))
        return False
