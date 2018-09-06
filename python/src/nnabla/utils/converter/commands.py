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

from .nnabla import NnpImporter, NnpExporter
from .nnablart import NnbExporter, CsrcExporter


def _import_file(args, ifiles):
    if len(ifiles) == 1 and os.path.splitext(ifiles[0])[1] == '.nnp':
        args.import_format = 'NNP'
    if len(ifiles) == 1 and os.path.splitext(ifiles[0])[1] == '.onnx':
        args.import_format = 'ONNX'
    if args.import_format == 'NNP':
        # Input file that has unsuported extension store into output nnp
        # archive or directory.
        return NnpImporter(*ifiles,
                           expand_network=not args.nnp_no_expand_network,
                           executor_index=args.nnp_import_executor_index).execute()
    elif args.import_format == 'ONNX':
        from .onnx import OnnxImporter
        return OnnxImporter(*ifiles).execute()
    return None


def _shrink_nnp(nnp, pos_start, pos_end):

    if len(nnp.protobuf.executor) != 1 or \
            len(nnp.protobuf.network) != 1:
        print('[ERROR] Please make only one network in nnp.')
        sys.exit(-1)
    from nnabla.utils import nnabla_pb2
    net = nnabla_pb2.NNablaProtoBuf().network.add()
    net.CopyFrom(nnp.protobuf.network[0])

    # Shrink network
    variables = {}
    net.ClearField('function')
    for i in range(pos_start, pos_end+1):
        f = nnp.protobuf.network[0].function[i]
        func = net.function.add()
        func.CopyFrom(f)
        for v in func.input:
            variables[v] = True
        for v in func.output:
            variables[v] = True

    net.ClearField('variable')
    for v in nnp.protobuf.network[0].variable:
        if v.name in variables:
            variables[v.name] = v.type
            var = net.variable.add()
            var.CopyFrom(v)

    # Shrink parameter
    params = []
    for param in nnp.protobuf.parameter:
        if param.variable_name in variables:
            p = nnabla_pb2.NNablaProtoBuf().parameter.add()
            p.CopyFrom(param)
            params.append(p)
    nnp.protobuf.ClearField('parameter')
    for p in params:
        param = nnp.protobuf.parameter.add()
        param.CopyFrom(p)

    # Shrink executor
    exe = nnabla_pb2.NNablaProtoBuf().executor.add()
    exe.CopyFrom(nnp.protobuf.executor[0])

    exe.ClearField('data_variable')
    for var in nnp.protobuf.executor[0].data_variable:
        if var.variable_name in variables:
            v = exe.data_variable.add()
            v.CopyFrom(var)

    # If data_variable is empty use input of first function instead.
    if len(exe.data_variable) == 0:
        for vname in nnp.protobuf.network[0].function[pos_start].input:
            if variables[vname] == 'Buffer':
                v = exe.data_variable.add()
                v.variable_name = vname

    exe.ClearField('generator_variable')
    for var in nnp.protobuf.executor[0].data_variable:
        if var.variable_name in variables:
            v = exe.data_variable.add()
            v.CopyFrom(var)

    exe.ClearField('output_variable')
    for var in nnp.protobuf.executor[0].output_variable:
        if var.variable_name in variables:
            v = exe.output_variable.add()
            v.CopyFrom(var)

    # If output_variable is empty use output of last function instead.
    if len(exe.output_variable) == 0:
        for vname in nnp.protobuf.network[0].function[pos_end].input:
            if variables[vname] == 'Buffer':
                v = exe.output_variable.add()
                v.variable_name = vname

    exe.ClearField('parameter_variable')
    for var in nnp.protobuf.executor[0].parameter_variable:
        if var.variable_name in variables:
            v = exe.parameter_variable.add()
            v.CopyFrom(var)

    nnp.protobuf.ClearField('network')
    n = nnp.protobuf.network.add()
    n.CopyFrom(net)
    nnp.protobuf.ClearField('executor')
    e = nnp.protobuf.executor.add()
    e.CopyFrom(exe)
    return nnp


def _export_from_nnp(args, nnp, output):
    output_ext = os.path.splitext(output)[1].lower()
    if (os.path.isdir(output) and args.export_format == 'NNP') \
            or output_ext == '.nnp':
        parameter_type = 'protobuf'
        if args.nnp_parameter_nntxt:
            parameter_type = 'included'
        elif args.nnp_parameter_h5:
            parameter_type = 'h5'
        if args.nnp_exclude_parameter:
            parameter_type = 'none'
        NnpExporter(nnp, args.batch_size, parameter_type).execute(output)

    elif output_ext == '.nnb':
        NnbExporter(nnp, args.batch_size).execute(
            output, None, args.settings, args.default_variable_type)

    elif os.path.isdir(output) and args.export_format == 'CSRC':
        CsrcExporter(nnp, args.batch_size).execute(output)

    elif output_ext == '.onnx':
        from .onnx import OnnxExporter
        OnnxExporter(nnp, args.batch_size).execute(output)
    else:
        print('Output file ({})'.format(output_ext) +
              ' is not supported or output directory does not exist.')
        return False
    return True


def convert_files(args, ifiles, output):
    nnp = _import_file(args, ifiles)
    if nnp is not None:
        if args.split is not None:
            if args.nnp_import_executor_index is None:
                print('[ERROR] "-S" needs "-E".')
                sys.exit(-1)
            network = None
            for n in nnp.protobuf.network:
                if n.name == nnp.protobuf.executor[0].network_name:
                    network = n
            ranges = []
            for srange in args.split.split(','):
                srange_s = srange.split('-')
                print(srange, srange_s, len(srange_s))
                pos_start = None
                pos_end = None
                if len(srange_s) == 2:
                    if srange_s[0] == '':
                        pos_start = 0
                    else:
                        pos_start = int(srange_s[0])
                    if srange_s[1] == '':
                        pos_end = len(network.function)-1
                    else:
                        pos_end = int(srange_s[1])
                    if pos_end < pos_start or pos_end > len(network.function)-1:
                        print('[ERROR] range must be in 0 to {}'.format(
                            len(network.function)-1))
                        sys.exit(-1)
                else:
                    print('[ERROR] range must be "x0-y0,x1-y1,..."')
                    sys.exit(-1)
                ranges.append((pos_start, pos_end))

            for pos_start, pos_end in ranges:
                print('   Shrink {} to {}.'.format(pos_start, pos_end))
                n, e = os.path.splitext(output)
                new_output = n + '_{}_{}'.format(pos_start, pos_end) + e
                print('    Output to [{}]'.format(new_output))
                _export_from_nnp(args, _shrink_nnp(
                    nnp, pos_start, pos_end), new_output)

        else:
            return _export_from_nnp(args, nnp, output)
    else:
        print('Import from [{}] failed.'.format(ifiles))
        return False


def _generate_nnb_template(args, nnp, output):
    NnbExporter(nnp, args.batch_size).execute(
        None, output, None, args.default_variable_type)
    return True


def nnb_template(args, ifiles, output):
    nnp = _import_file(args, ifiles)
    if nnp is not None:
        return _generate_nnb_template(args, nnp, output)
    else:
        print('Import from [{}] failed.'.format(ifiles))
        return False


def _dump_protobuf(args, proto, prefix, depth):
    if args.dump_verbose:
        if depth >= 0 and len(prefix) >= depth:
            print('{} ...'.format(':'.join(prefix)))
            return
        for desc, field in proto.ListFields():
            if isinstance(field, (int, float, complex, str)):
                print('{}:{}: {}'.format(':'.join(prefix), desc.name, field))
            elif isinstance(field, collections.Iterable):
                print('{} has {} {}(s).'.format(
                    ':'.join(prefix), len(field), desc.name))
                for n, f in enumerate(field[:args.dump_limit]):
                    if isinstance(f, (int, float, complex, str)):
                        print('{}:{}[{}]: {}'.format(
                            ':'.join(prefix), desc.name, n, f))
                    else:
                        if depth < 0 or depth > len(prefix)+1:
                            _dump_protobuf(
                                args, f, prefix + ['{}[{}]'.format(desc.name, n)], depth)
            else:
                _dump_protobuf(args, field, prefix + [desc.name], depth)
    else:
        params = {}
        for par in proto.parameter:
            params[par.variable_name] = [x for x in par.shape.dim]

        nets = {}
        for net in proto.network:
            ninfo = {'variables': {}, 'functions': []}
            for var in net.variable:
                if var.type == 'Parameter':
                    if var.name not in params:
                        print('[ERROR] Parameter [{}] in network[{}] not found.'.format(
                            var.name, net.name))
                        print('        ' +
                              'Make sure that you do not forget to read parameter file.')
                        print('        ' +
                              'Otherwise it should be expander problem. Please report us.')
                        sys.exit(-1)

                ninfo['variables'][var.name] = {
                    'type': var.type,
                    'shape': [x for x in var.shape.dim]}

            for func in net.function:
                ninfo['functions'].append(func)

            nets[net.name] = ninfo

        def _dump_network_arg(prefix, name, var):
            for i, v in enumerate(var):
                shape = ' - '
                if v.variable_name in net['variables']:
                    shape = net['variables'][v.variable_name]['shape']
                print('{}{} variable[{}]:'.format(prefix, name, i) +
                      ' Name:{:30}'.format(v.variable_name) +
                      ' Shape:{}'.format(shape))

        def _dump_network(prefix, net):
            if args.dump_functions:
                for i, f in enumerate(net['functions']):
                    func_prefix = '{}  Function[{:^5}]: '.format(prefix, i)
                    print('{}Type: {:20} Name: {}'.format(
                        func_prefix, f.type, f.name))
                    if args.dump_variables:
                        for j, v in enumerate(f.input):
                            print('{}  Input{}: Name: {:20} Shape: {}'.format(
                                func_prefix, j, v, net['variables'][v]['shape']))
                        for j, v in enumerate(f.output):
                            print('{} Output{}: Name: {:20} Shape: {}'.format(
                                func_prefix, j, v, net['variables'][v]['shape']))

        for i, opt in enumerate(proto.optimizer):
            net = nets[opt.network_name]
            prefix = '  Optimizer[{}]: '.format(i)
            print('{}{}'.format(prefix, opt.name))

            _dump_network_arg(prefix, ' (In) Data     ', opt.data_variable)
            _dump_network_arg(prefix, ' (In) Generator',
                              opt.generator_variable)
            _dump_network_arg(prefix, ' (Out)Loss     ', opt.loss_variable)
            _dump_network(prefix, net)

        for i, mon in enumerate(proto.monitor):
            net = nets[mon.network_name]
            prefix = '  Monitor  [{}]: '.format(i)
            print('{}{}'.format(prefix, mon.name))
            _dump_network_arg(prefix, ' (In) Data     ', mon.data_variable)
            _dump_network_arg(prefix, ' (In) Generator',
                              mon.generator_variable)
            _dump_network_arg(prefix, ' (Out)Monitor  ', mon.monitor_variable)
            _dump_network(prefix, net)

        for i, exe in enumerate(proto.executor):
            net = nets[exe.network_name]
            prefix = '  Executor [{}]: '.format(i)
            print('{}{}'.format(prefix, exe.name))
            _dump_network_arg(prefix, ' (In) Data     ', exe.data_variable)
            _dump_network_arg(prefix, ' (In) Generator',
                              exe.generator_variable)
            _dump_network_arg(prefix, ' (Out)Loss     ', exe.loss_variable)
            _dump_network_arg(prefix, ' (Out)Output   ', exe.output_variable)
            _dump_network(prefix, net)


def _dump_nnp(args, nnp):
    _dump_protobuf(args, nnp.protobuf, [args.import_format], -1)
    return True


def dump_files(args, ifiles):
    nnp = _import_file(args, ifiles)
    if nnp is not None:
        return _dump_nnp(args, nnp)
    else:
        print('Import from [{}] failed.'.format(ifiles))
        return False
