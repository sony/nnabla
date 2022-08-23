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
import math
import struct

import nnabla.utils.converter
import numpy as np
import yaml

from .resolver import Resolver
from .utils import create_nnabart_info
from .utils import preprocess_for_exporter
from .utils import revise_buffer_size

NN_BINARY_FORMAT_VERSION = 2


class Nnb:
    '''
    Nnb is only used as namespace
    '''
    NN_DATA_TYPE_FLOAT, NN_DATA_TYPE_INT16, NN_DATA_TYPE_INT8, NN_DATA_TYPE_SIGN = range(
        4)
    from_type_name = {
        'FLOAT32': NN_DATA_TYPE_FLOAT,
        'FIXED16': NN_DATA_TYPE_INT16,
        'FIXED8': NN_DATA_TYPE_INT8,
        'SIGN': NN_DATA_TYPE_SIGN
    }
    fp_pos_max = {NN_DATA_TYPE_INT16: 15, NN_DATA_TYPE_INT8: 7}


class NnbExporter:
    def _align(self, size):
        return int(math.ceil(float(size) / 4) * 4)

    def _alloc(self, size=-1, data=b''):
        size = len(data) if size < 0 else size
        index = len(self._memory_index)
        pointer = sum(self._memory_index)
        self._memory_index.append(len(self._memory_data))
        assert (len(data) <= size)
        self._memory_data += data
        self._memory_data += b'\0' * (self._align(len(data)) - len(data))
        return (index, pointer)

    def __init__(self, nnp, batch_size, nnb_version=NN_BINARY_FORMAT_VERSION,
                 api_level=-1):
        nnp = Resolver(nnp).execute()
        self._info = create_nnabart_info(nnp, batch_size)
        self._api_level_info = nnabla.utils.converter.get_api_level_info()
        self._nnb_version = nnb_version
        preprocess_for_exporter(self._info, 'NNB')

        self._List = collections.namedtuple('List', ('size', 'list_index'))

        self._memory_index = []
        self._memory_data = b''
        self._api_level_info.set_api_level(api_level)

    @staticmethod
    def __compute_int_bit_num(param_array):
        abs_array = np.abs(param_array)
        max_abs = abs_array.max()
        if max_abs >= 1:
            max_idx = abs_array.argmax()
            max_log2 = np.log2(max_abs)
            if max_log2.is_integer() and param_array[max_idx] > 0:
                int_bit_num = int(max_log2) + 2  # almost impossible
            else:
                int_bit_num = int(np.ceil(max_log2)) + 1
        else:
            int_bit_num = 1  # 1 is needed to represent sign
        return int_bit_num

    def execute(self, nnb_output_filename, settings_template_filename, settings_filename, default_type):
        settings = collections.OrderedDict()
        if settings_filename is not None and len(settings_filename) == 1:
            settings = nnabla.utils.converter.load_yaml_ordered(
                open(settings_filename[0]))
        if 'functions' not in settings:
            settings['functions'] = collections.OrderedDict()
        if 'variables' not in settings:
            settings['variables'] = collections.OrderedDict()

        ####################################################################
        # Version
        api_level = self._api_level_info.get_current_level()

        ####################################################################
        # Variables name index
        vindexes_by_name = {}
        for n, v in enumerate(self._info._network.variable):
            vindexes_by_name[v.name] = n

        ####################################################################
        # Inputs
        input_list = [vindexes_by_name[i]
                      for i in self._info._input_variables]
        index, pointer = self._alloc(data=struct.pack(
            '{}I'.format(len(input_list)), *input_list))
        inputs = self._List(len(input_list), index)

        ####################################################################
        # Outputs
        output_list = [vindexes_by_name[i]
                       for i in self._info._output_variables]
        index, pointer = self._alloc(data=struct.pack(
            '{}I'.format(len(output_list)), *output_list))
        outputs = self._List(len(output_list), index)

        ####################################################################
        # revise buffer size by bytes instead of data item.
        if self._nnb_version > NN_BINARY_FORMAT_VERSION:
            revise_buffer_size(self._info, settings)

        ####################################################################
        # make 2 data to save Variable Buffers in inference
        from .save_variable_buffer import save_variable_buffer
        actual_buf_sizes, vidx_to_abidx = save_variable_buffer(self._info)

        ####################################################################
        # Variable buffers
        blist = actual_buf_sizes
        index, pointer = self._alloc(
            data=struct.pack('{}I'.format(len(blist)), *blist))
        buffers = self._List(len(blist), index)

        ####################################################################
        # Variables
        self._Variable = collections.namedtuple(
            'Variable', ('id', 'shape', 'type', 'fp_pos', 'data_index'))
        vindexes = []
        for n, v in enumerate(self._info._network.variable):
            var = self._Variable
            var.id = n

            # set var.shape and store into NNB
            shape = [
                x if x >= 0 else self._info._batch_size for x in v.shape.dim]
            index, pointer = self._alloc(
                data=struct.pack('{}I'.format(len(shape)), *shape))
            var.shape = self._List(len(shape), index)

            # parse a type option in YAML given via -settings
            if v.name not in settings['variables']:
                settings['variables'][v.name] = default_type[0]
            type_option = settings['variables'][v.name]
            opt_list = type_option.split('_')
            type_name = opt_list[0]
            fp_pos = int(opt_list[1]) if len(opt_list) == 2 else None

            # set var.type, var.data_index, and var.fp_pos in this paragraph
            var.type = Nnb.from_type_name[type_name]
            if v.name in self._info._generator_variables:
                data = self._info._generator_variables[v.name]
                data = data.flatten().tolist()
                fmt = '{}f'.format(len(data))
                raw_data = struct.pack(fmt, *data)
                index, pointer = self._alloc(data=raw_data)
                var.data_index = index
            elif v.type == 'Parameter':
                # store parameter into NNB
                array = np.array(self._info._parameters[v.name].data)
                if type_name == 'FLOAT32':
                    fmt_base = '{}f'
                elif type_name == 'SIGN':
                    array = array.astype(np.uint8)
                    array[array == 255] = 0
                    array = array.reshape(-1, 8)
                    for i in range(array.shape[0]):
                        array[i] = array[i][::-1]
                    array = array.flatten()
                    fmt_base = '{}B'
                    array = np.packbits(array)
                else:  # type_name == 'FIXED16' or type_name == 'FIXED8'
                    fmt_base = '{}h' if type_name == 'FIXED16' else '{}b'
                    # if fp_pos is not specified, compute it looking at its distribution
                    if fp_pos is None:
                        int_bit_num = NnbExporter.__compute_int_bit_num(array)
                        fp_pos = (Nnb.fp_pos_max[var.type] + 1) - int_bit_num
                    else:
                        pass  # do nothing
                    # convert float to fixed point values
                    scale = 1 << fp_pos
                    array = np.round(array * scale).astype(int)
                    if type_name == 'FIXED16':
                        array = np.clip(array, -0x7fff - 1, 0x7fff)
                    elif type_name == 'FIXED8':
                        array = np.clip(array, -128, 127)
                fmt = fmt_base.format(len(array))
                data = struct.pack(fmt, *array)
                index, pointer = self._alloc(data=data)
                var.data_index = index
            elif v.type == 'Buffer':
                if var.type == Nnb.NN_DATA_TYPE_SIGN:
                    raise ValueError(
                        'Unsupport SIGN type for Buffer Variable.')
                # check fp_pos
                if var.type != Nnb.NN_DATA_TYPE_FLOAT and fp_pos is None:
                    msg = 'fp_pos must be specified for Buffer Variable'
                    raise ValueError(msg)
                # FIXME: remove the following workaround
                if n in vidx_to_abidx:
                    # n which is NOT in vidx_to_abidx can appear
                    # since NnpExpander doesn't handle --nnp-expand-network correctly
                    var.data_index = (vidx_to_abidx[n] + 1) * -1
                else:
                    # this var doesn't make sense, but add  it
                    # so that nn_network_t::variables::size is conserved
                    var.data_index = -1
            # check fp_pos and set var.fp_pos
            if var.type == Nnb.NN_DATA_TYPE_INT16 or var.type == Nnb.NN_DATA_TYPE_INT8:
                if 0 <= fp_pos or fp_pos <= Nnb.fp_pos_max[var.type]:
                    var.fp_pos = fp_pos
                else:
                    raise ValueError('invalid fp_pos was given')
            else:
                var.fp_pos = 0

            variable = struct.pack('IiIBi',
                                   var.id,
                                   var.shape.size, var.shape.list_index,
                                   ((var.fp_pos & 0xf) << 4 | (var.type & 0xf)),
                                   var.data_index)
            index, pointer = self._alloc(data=variable)
            vindexes.append(index)

        index, pointer = self._alloc(data=struct.pack(
            '{}I'.format(len(vindexes)), *vindexes))
        variables = self._List(len(vindexes), index)

        ####################################################################
        # Functions
        findexes = []
        for n, f in enumerate(self._info._network.function):
            if f.type not in self._api_level_info.get_function_list():
                raise ValueError("{}() is not supported in current API Level(={}).".format(
                    f.type, self._api_level_info.get_current_level()))

            function_data = struct.pack(
                'H', self._api_level_info.get_func_id(f.type))

            # Default function implementation is 0(float)
            if f.name not in settings['functions']:
                settings['functions'][f.name] = collections.OrderedDict()
                settings['functions'][f.name]['implement'] = 0

            function_data += struct.pack('H',
                                         settings['functions'][f.name]['implement'])

            finfo = self._info._function_info[f.type]

            finputs = [vindexes_by_name[i] for i in f.input]
            index, pointer = self._alloc(data=struct.pack(
                '{}I'.format(len(finputs)), *finputs))
            function_data += struct.pack('iI', len(finputs), index)

            foutputs = [vindexes_by_name[o] for o in f.output]
            index, pointer = self._alloc(data=struct.pack(
                '{}I'.format(len(foutputs)), *foutputs))
            function_data += struct.pack('iI', len(foutputs), index)

            argcode = self._api_level_info.get_argument_code(f.type)
            argcode_pos = 0
            if 'arguments' in finfo and len(finfo['arguments']) > 0:
                argfmt = ''
                values = []
                for an, arg in finfo['arguments'].items():
                    val = eval('f.{}_param.{}'.format(
                        finfo['snake_name'], an))
                    arg_type_id = nnabla.utils.converter.type_to_pack_format(
                        arg['type'])
                    if argcode_pos >= len(argcode):
                        # ommit the parameter that is not supported
                        # we only down-version by omitting the tail-appended parameters.
                        print("{}.{} is omitted for lower API Level:{}".format(f.type, an,
                                                                               self._api_level_info.get_current_level()))
                        continue
                    else:
                        # If argument type is changed, this function will be
                        # unable to down-version.
                        if argcode[argcode_pos:argcode_pos + len(arg_type_id)] != arg_type_id:
                            raise ValueError("{} is not supported by API Level:{}."
                                             .format(self._api_level_info.get_func_uniq_name(f.type),
                                                     self._api_level_info.get_current_level()))
                        argcode_pos += len(arg_type_id)
                    argfmt += arg_type_id
                    if arg['type'] == 'bool':
                        values.append(val)
                    elif arg['type'] == 'double' or arg['type'] == 'float':
                        values.append(val)
                    elif arg['type'] == 'int64':
                        values.append(val)
                    elif arg['type'] == 'repeated int64':
                        index, pointer = self._alloc(
                            data=struct.pack('{}i'.format(len(val)), *val))
                        values.append(len(val))
                        values.append(index)
                    elif arg['type'] == 'Shape':
                        index, pointer = self._alloc(data=struct.pack(
                            '{}i'.format(len(val.dim)), *val.dim))
                        values.append(len(val.dim))
                        values.append(index)
                    elif arg['type'] == 'string':
                        val = arg['available_values'].index(val)
                        values.append(val)
                function_data += struct.pack(argfmt, *values)
            else:
                # Check if old version requires argument.
                # If it is true, down-version is not allowed.
                if argcode:
                    raise ValueError("{} is not supported by API Level:{}."
                                     .format(self._api_level_info.get_func_uniq_name(f.type),
                                             self._api_level_info.get_current_level()))

            index, pointer = self._alloc(data=function_data)
            findexes.append(index)

        index, pointer = self._alloc(data=struct.pack(
            '{}I'.format(len(findexes)), *findexes))
        functions = self._List(len(findexes), index)

        network = struct.pack('IIiIiIiIiIiIII',
                              self._nnb_version,
                              api_level,
                              buffers.size,
                              buffers.list_index,
                              variables.size,
                              variables.list_index,
                              functions.size,
                              functions.list_index,
                              inputs.size,
                              inputs.list_index,
                              outputs.size,
                              outputs.list_index,
                              len(self._memory_index),
                              len(self._memory_data))
        memory = struct.pack('{}I'.format(
            len(self._memory_index)), *self._memory_index) + self._memory_data

        if settings_template_filename is not None:
            with open(settings_template_filename, 'w') as f:
                f.write(yaml.dump(settings, default_flow_style=False))

        if nnb_output_filename is not None:
            with open(nnb_output_filename, 'wb') as f:
                f.write(network + memory)
