# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import collections
import os
import sys

from .nnabla import NnpImporter, NnpExporter
from .nnablart import NnbExporter, CsrcExporter
from .utils import func_set_import_nnp, \
    func_set_import_onnx_config, \
    func_set_import_config, \
    func_set_nnabla_support, \
    func_set_onnx_support, \
    func_set_nnabla_version_decorate
from nnabla.logger import logger


def _import_file(args, ifiles):
    if args.import_format == 'NNP':
        # Input file that has unsupported extension store into output nnp
        # archive or directory.
        return NnpImporter(*ifiles,
                           expand_network=not args.nnp_no_expand_network,
                           executor_index=args.nnp_import_executor_index).execute()

    elif args.import_format == 'ONNX':
        from .onnx import OnnxImporter
        return OnnxImporter(*ifiles).execute()

    elif args.import_format == 'TF_PB' or \
            args.import_format == 'TF_CKPT_V1' or \
            args.import_format == "TF_CKPT_V2" or \
            args.import_format == "SAVED_MODEL":
        from .tensorflow import TensorflowImporter
        return TensorflowImporter(*ifiles, tf_format=args.import_format, outputs=args.outputs, inputs=args.inputs).execute()
    elif args.import_format == 'TFLITE':
        from .tflite import TFLiteImporter
        return TFLiteImporter(*ifiles).execute()
    return None


def _shrink_nnp(nnp, pos_start, pos_end):
    if len(nnp.protobuf.executor) != 1 or \
            len(nnp.protobuf.network) != 1:
        print('[ERROR] Please make only one network in nnp.')
        sys.exit(-1)
    from nnabla.utils import nnabla_pb2

    class _nnp:
        pass
    _nnp.protobuf = nnabla_pb2.NNablaProtoBuf()
    _nnp.other_files = nnp.other_files
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
    for p in params:
        param = _nnp.protobuf.parameter.add()
        param.CopyFrom(p)

    # Shrink executor
    exe = nnabla_pb2.NNablaProtoBuf().executor.add()
    exe.CopyFrom(nnp.protobuf.executor[0])

    exe.ClearField('data_variable')
    for vname in nnp.protobuf.network[0].function[pos_start].input:
        if variables[vname] == 'Buffer':
            v = exe.data_variable.add()
            v.variable_name = vname

    exe.ClearField('generator_variable')
    for var in nnp.protobuf.executor[0].generator_variable:
        if var.variable_name in variables:
            v = exe.generator_variable.add()
            v.CopyFrom(var)

    exe.ClearField('output_variable')
    for vname in nnp.protobuf.network[0].function[pos_end].output:
        if variables[vname] == 'Buffer':
            v = exe.output_variable.add()
            v.variable_name = vname

    exe.ClearField('parameter_variable')
    for var in nnp.protobuf.executor[0].parameter_variable:
        if var.variable_name in variables:
            v = exe.parameter_variable.add()
            v.CopyFrom(var)

    n = _nnp.protobuf.network.add()
    n.CopyFrom(net)
    e = _nnp.protobuf.executor.add()
    e.CopyFrom(exe)
    return _nnp


def _export_from_nnp(args, nnp, output):
    if args.export_format == 'NNP':
        parameter_type = 'protobuf'
        if args.nnp_parameter_nntxt:
            parameter_type = 'included'
        elif args.nnp_parameter_h5:
            parameter_type = 'h5'
        if args.nnp_exclude_parameter:
            parameter_type = 'none'
        nnp = func_set_nnabla_version_decorate(nnp, args.nnp_version)
        NnpExporter(nnp, args.batch_size, parameter_type).execute(output)
        logger.info(f"Converting: {output} successfully!")

    elif args.export_format == 'NNB':
        if args.batch_size < 0:
            print('NNB: Batch size adjust to 1.')
            print('NNB: If you want to use with other size use `-b` option.')
            args.batch_size = 1
        if args.define_version and args.define_version.startswith('nnb_'):
            nnb_version = int(args.define_version.split("_")[1])
            NnbExporter(nnp, args.batch_size, nnb_version=nnb_version, api_level=args.api).execute(
                output, None, args.settings, args.default_variable_type)
        else:
            NnbExporter(nnp, args.batch_size, api_level=args.api).execute(
                output, None, args.settings, args.default_variable_type)

    elif args.export_format == 'CSRC':
        if args.batch_size < 0:
            print('CSRC: Batch size adjust to 1.')
            print('CSRC: If you want to use with other size use `-b` option.')
            args.batch_size = 1
        CsrcExporter(nnp, args.batch_size).execute(output)

    elif args.export_format == 'ONNX':
        from .onnx import OnnxExporter
        if args.define_version and args.define_version.startswith('opset_'):
            opset = args.define_version.split("_")[1]
            OnnxExporter(nnp, args.batch_size, opset=opset).execute(output)
        else:
            OnnxExporter(nnp, args.batch_size).execute(output)
    elif args.export_format == 'SAVED_MODEL' or args.export_format == 'TF_PB':
        from .tensorflow import TensorflowExporter
        TensorflowExporter(nnp, args.batch_size,
                           model_format=args.export_format).execute(output)
    elif args.export_format == 'TFLITE':
        from .tflite import TFLiteExporter
        TFLiteExporter(nnp, args.batch_size,
                       channel_last=args.channel_last, quantization=args.quantization,
                       dataset=args.dataset).execute(output)
    else:
        print('Output file ({})'.format(args.export_format) +
              ' is not supported or output directory does not exist.')
        return False
    return True


def _need_split(nnp, args, supported_set):
    if args.split is not None:
        if args.nnp_import_executor_index is None:
            print('[ERROR] "-S" needs "-E".')
            sys.exit(-1)
        return True

    if not args.config:
        return False

    if func_set_import_nnp(nnp) <= supported_set:
        return False

    return True


def _get_split_ranges(nnp, args, supported_set):
    def get_ranges_from_param(split_spec):
        ranges = []
        for srange in split_spec.split(','):
            srange_s = srange.split('-')
            if len(srange_s) == 2:
                if srange_s[0] == '':
                    pos_start = 0
                else:
                    pos_start = int(srange_s[0])
                if srange_s[1] == '':
                    pos_end = len(network.function) - 1
                else:
                    pos_end = int(srange_s[1])
                if pos_end < pos_start or pos_end > len(network.function) - 1:
                    print('[ERROR] range must be in 0 to {}'.format(
                        len(network.function) - 1))
                    sys.exit(-1)
            else:
                print('[ERROR] range must be "x0-y0,x1-y1,..."')
                sys.exit(-1)
            ranges.append((pos_start, pos_end))
        return ranges

    def get_ranges_from_func_set(support_set):
        pos_start = 0
        pos_end = 0
        ranges = []
        for pos, func in enumerate(network.function):
            if func.type in support_set:
                pos_end = pos
            else:
                if pos_end >= pos_start:
                    ranges.append((pos_start, pos_end))
                pos_start = pos + 1
        if pos_end >= pos_start:
            ranges.append((pos_start, pos_end))
        return ranges

    network = None
    for n in nnp.protobuf.network:
        if n.name == nnp.protobuf.executor[0].network_name:
            network = n
    if args.split:
        return get_ranges_from_param(args.split)
    return get_ranges_from_func_set(supported_set)


def convert_files(args, ifiles, output):
    nnp = _import_file(args, ifiles)
    if nnp is not None:
        network_name = nnp.protobuf.executor[0].network_name
        if args.export_format == 'ONNX':
            if args.config:
                support_set = func_set_onnx_support() & \
                              func_set_import_onnx_config(args.config)
            else:
                support_set = func_set_onnx_support()
        elif args.export_format == 'NNB':
            if args.config:
                support_set = func_set_import_config(args.config)
            else:
                support_set = func_set_nnabla_support()
        else:
            if args.config:
                support_set = func_set_import_config(args.config)
            else:
                if args.nnp_version is not None:
                    version_spec = (nnp, args.nnp_version)
                else:
                    version_spec = None
                support_set = func_set_nnabla_support(version_spec)
        if _need_split(nnp, args, support_set):
            for _net in nnp.protobuf.network[:]:
                if _net.name != network_name:
                    nnp.protobuf.network.remove(_net)
            ranges = _get_split_ranges(nnp, args, support_set)
            nnb_info = collections.OrderedDict()
            for pos_start, pos_end in ranges:
                print('   Shrink {} to {}.'.format(pos_start, pos_end))
                n, e = os.path.splitext(output)
                new_output = n + '_{}_{}'.format(pos_start, pos_end) + e
                print('    Output to [{}]'.format(new_output))
                _nnp = _shrink_nnp(nnp, pos_start, pos_end)
                nnb_info[new_output] = collections.OrderedDict()
                nnb_info[new_output]['input'] = []
                nnb_info[new_output]['output'] = []
                i_var = []
                o_var = []
                for var in _nnp.protobuf.executor[0].data_variable:
                    i_var.append(var.variable_name)
                for var in _nnp.protobuf.executor[0].output_variable:
                    o_var.append(var.variable_name)
                for var in _nnp.protobuf.network[0].variable:
                    if var.name in i_var:
                        _info = collections.OrderedDict()
                        _info['name'] = var.name
                        _info['shape'] = '({})'.format(
                            ', '.join(str(i) for i in var.shape.dim))
                        nnb_info[new_output]['input'].append(_info)
                    if var.name in o_var:
                        _info = collections.OrderedDict()
                        _info['name'] = var.name
                        _info['shape'] = '({})'.format(
                            ', '.join(str(i) for i in var.shape.dim))
                        nnb_info[new_output]['output'].append(_info)
                _export_from_nnp(args, _nnp, new_output)
            import yaml
            print(yaml.dump(nnb_info, default_flow_style=False))
        else:
            return _export_from_nnp(args, nnp, output)
    else:
        print('Import from {} failed.'.format(ifiles))
        return False


def _generate_nnb_template(args, nnp, output):
    NnbExporter(nnp, args.batch_size, api_level=args.api).execute(
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
        if 0 <= depth <= len(prefix):
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
            if args.dump_variable_name:
                if args.dump_variable_name in net['variables']:
                    v = args.dump_variable_name
                    print('Variable Name: {:20} Shape: {}'.format(
                        v, net['variables'][v]['shape']))
                else:
                    print('DUMP ERROR: variable {} not found.'.format(
                        args.dump_variable_name))
                return
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
