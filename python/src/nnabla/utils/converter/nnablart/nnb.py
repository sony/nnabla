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
import math
import numpy as np
import struct
import yaml

import nnabla.utils.nnabla_pb2 as nnabla_pb2
import nnabla.utils.converter

from .utils import create_nnabart_info


class Nnb:
    '''
    Nnb is only used as namespace
    '''
    NN_DATA_TYPE_FLOAT, NN_DATA_TYPE_INT16, NN_DATA_TYPE_INT8, NN_DATA_TYPE_SIGN = range(
        4)


class NnbExporter:
    def _align(self, size):
        return int(math.ceil(size / 4) * 4)

    def _alloc(self, size=-1, data=b''):
        size = len(data) if size < 0 else size
        index = len(self._memory_index)
        pointer = sum(self._memory_index)
        self._memory_index.append(len(self._memory_data))
        assert(len(data) <= size)
        self._memory_data += data
        self._memory_data += b'\0' * (self._align(len(data)) - len(data))
        return (index, pointer)

    def __init__(self, nnp, batch_size):
        self._info = create_nnabart_info(nnp, batch_size)

        self._List = collections.namedtuple('List', ('size', 'list_index'))

        self._memory_index = []
        self._memory_data = b''

        self._argument_formats = {}
        for fn, func in self._info._function_info.items():
            if 'arguments' in func and len(func['arguments']) > 0:
                argfmt = ''
                for an, arg in func['arguments'].items():
                    argfmt += nnabla.utils.converter.type_to_pack_format(
                        arg['type'])
                self._argument_formats[fn] = argfmt

    def export(self, nnb_output_filename, settings_template_filename, settings_filename, default_type):
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
        version = nnabla.utils.converter.get_category_info_version()

        ####################################################################
        # Varibles name index
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
        # make 2 data to save Variable Buffers in inference
        from .save_variable_buffer import save_variable_buffer
        actual_buf_sizes, vidx_to_abidx = save_variable_buffer(self._info)

        ####################################################################
        # Varible buffers
        blist = actual_buf_sizes
        index, pointer = self._alloc(
            data=struct.pack('{}I'.format(len(blist)), *blist))
        buffers = self._List(len(blist), index)

        ####################################################################
        # Varibles
        self._Variable = collections.namedtuple(
            'Variable', ('id', 'shape', 'type', 'fp_pos', 'data_index'))
        vindexes = []
        for n, v in enumerate(self._info._network.variable):
            var = self._Variable
            var.id = n

            shape = [
                x if x >= 0 else self._info._batch_size for x in v.shape.dim]
            index, pointer = self._alloc(
                data=struct.pack('{}I'.format(len(shape)), *shape))
            var.shape = self._List(len(shape), index)

            var.type = 0  # NN_DATA_TYPE_FLOAT
            var.fp_pos = 0

            if v.name not in settings['variables']:
                settings['variables'][v.name] = default_type[0]

            if v.type == 'Parameter':
                param = self._info._parameters[v.name]
                param_data = list(param.data)
                v_name = settings['variables'][v.name]
                if v_name == 'FLOAT32':
                    data = struct.pack('{}f'.format(
                        len(param_data)), *param_data)
                    var.type = Nnb.NN_DATA_TYPE_FLOAT
                elif v_name.startswith('FIXED16'):
                    fixed16_desc = v_name.split('_')
                    if (len(fixed16_desc) == 2) and int(fixed16_desc[1]) <= 15:
                        var.fp_pos = int(fixed16_desc[1])
                    scale = 1 << var.fp_pos
                    fixed16_n_data = [int(round(x * scale))
                                      for x in param_data]
                    data = struct.pack('{}h'.format(
                        len(fixed16_n_data)), *fixed16_n_data)
                    var.type = Nnb.NN_DATA_TYPE_INT16
                elif v_name.startswith('FIXED8'):
                    fixed8_desc = v_name.split('_')
                    if (len(fixed8_desc) == 2) and int(fixed8_desc[1]) <= 7:
                        var.fp_pos = int(fixed8_desc[1])
                    scale = 1 << var.fp_pos
                    fixed8_n_data = [int(round(x * scale)) for x in param_data]
                    data = struct.pack('{}b'.format(
                        len(fixed8_n_data)), *fixed8_n_data)
                    var.type = Nnb.NN_DATA_TYPE_INT8

                index, pointer = self._alloc(data=data)
                var.data_index = index
            elif v.type == 'Buffer':
                # FIXME: remove the following workaround
                if n in vidx_to_abidx:
                    # n which is NOT in vidx_to_abidx can appear
                    # since NnpExpander doesn't handle --nnp-expand-network correctly
                    var.data_index = (vidx_to_abidx[n] + 1) * -1
                else:
                    # this var doesn't make sense, but add  it
                    # so that nn_network_t::variables::size is conserved
                    var.data_index = -1

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

            function_data = struct.pack(
                'I', list(self._info._function_info.keys()).index(f.type))

            # Default function implementation is 0(float)
            if f.name not in settings['functions']:
                settings['functions'][f.name] = collections.OrderedDict()
                settings['functions'][f.name]['implement'] = 0

            function_data += struct.pack('I',
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

            if 'arguments' in finfo and len(finfo['arguments']) > 0:
                argfmt = ''
                values = []
                for an, arg in finfo['arguments'].items():
                    val = eval('f.{}_param.{}'.format(
                        finfo['snake_name'], an))

                    argfmt += nnabla.utils.converter.type_to_pack_format(
                        arg['type'])
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

            index, pointer = self._alloc(data=function_data)
            findexes.append(index)

        index, pointer = self._alloc(data=struct.pack(
            '{}I'.format(len(findexes)), *findexes))
        functions = self._List(len(findexes), index)

        network = struct.pack('IiIiIiIiIiIII',
                              version,
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

        if settings_template_filename is not None and len(settings_template_filename) == 1:
            with open(settings_template_filename[0], 'w') as f:
                f.write(yaml.dump(settings, default_flow_style=False))

        with open(nnb_output_filename, 'wb') as f:
            f.write(network + memory)
