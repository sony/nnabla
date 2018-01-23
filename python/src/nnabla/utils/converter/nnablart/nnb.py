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
import os
import re
import struct

import nnabla.utils.nnabla_pb2 as nnabla_pb2
import nnabla.utils.converter

from .utils import create_nnabart_info


class NnbExporter:
    def _align(self, size):
        return math.ceil(size / 4) * 4

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
            if 'argument' in func:
                argfmt = ''
                for an, arg in func['argument'].items():
                    if arg['Type'] == 'bool':
                        argfmt += 'B'
                    elif arg['Type'] == 'double' or arg['Type'] == 'float':
                        argfmt += 'f'
                    elif arg['Type'] == 'int64':
                        argfmt += 'i'
                    elif arg['Type'] == 'repeated int64' or arg['Type'] == 'Shape':
                        argfmt += 'iI'
                    elif arg['Type'] == 'string':
                        argfmt += 'i'
                self._argument_formats[fn] = argfmt

    def export(self, *args):
        if len(args) == 1:
            ####################################################################
            # Version
            version = nnabla.utils.converter.get_category_info_version()

            ####################################################################
            # Varible buffers
            blist = list(self._info._variable_buffer_size.values())
            index, pointer = self._alloc(
                data=struct.pack('{}I'.format(len(blist)), *blist))
            buffers = self._List(len(blist), index)

            ####################################################################
            # Varibles
            self._Variable = collections.namedtuple(
                'Variable', ('id', 'shape', 'type', 'fp_pos', 'data_index'))
            vindexes_by_name = {}
            vindexes = []
            for n, v in enumerate(self._info._network.variable):
                var = self._Variable
                var.id = n
                vindexes_by_name[v.name] = n
                shape = [
                    x if x >= 0 else self._info._batch_size for x in v.shape.dim]
                index, pointer = self._alloc(
                    data=struct.pack('{}I'.format(len(shape)), *shape))
                var.shape = self._List(len(shape), index)

                var.type = 0  # NN_DATA_TYPE_FLOAT
                var.fp_pos = 0

                if v.type == 'Parameter':
                    param = self._info._parameters[v.name]
                    param_data = list(param.data)
                    data = struct.pack('{}f'.format(
                        len(param_data)), *param_data)
                    index, pointer = self._alloc(data=data)
                    var.data_index = index
                elif v.type == 'Buffer':
                    var.data_index = (
                        self._info._buffer_ids[n] + 1) * -1

                variable = struct.pack('IiIBi',
                                       var.id,
                                       var.shape.size, var.shape.list_index,
                                       (var.type & 0xf << 4) | (
                                           var.fp_pos & 0xf),
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

                finfo = self._info._function_info[f.type]

                inputs = [vindexes_by_name[i] for i in f.input]
                index, pointer = self._alloc(data=struct.pack(
                    '{}I'.format(len(inputs)), *inputs))
                function_data += struct.pack('iI', len(inputs), index)

                outputs = [vindexes_by_name[o] for o in f.output]
                index, pointer = self._alloc(data=struct.pack(
                    '{}I'.format(len(outputs)), *outputs))
                function_data += struct.pack('iI', len(outputs), index)

                if 'argument' in finfo:
                    argfmt = ''
                    values = []
                    for an, arg in finfo['argument'].items():
                        val = eval('f.{}_param.{}'.format(
                            finfo['snakecase_name'], an))
                        if arg['Type'] == 'bool':
                            argfmt += 'B'
                            values.append(val)
                        elif arg['Type'] == 'double' or arg['Type'] == 'float':
                            argfmt += 'f'
                            values.append(val)
                        elif arg['Type'] == 'int64':
                            argfmt += 'i'
                            values.append(val)
                        elif arg['Type'] == 'repeated int64':
                            index, pointer = self._alloc(
                                data=struct.pack('{}i'.format(len(val)), *val))
                            values.append(len(val))
                            values.append(index)
                            argfmt += 'iI'
                        elif arg['Type'] == 'Shape':
                            argfmt += 'iI'
                            index, pointer = self._alloc(data=struct.pack(
                                '{}i'.format(len(val.dim)), *val.dim))
                            values.append(len(val.dim))
                            values.append(index)
                        elif arg['Type'] == 'string':
                            argfmt += 'i'
                            val = arg['TypeSelection'].index(val)
                            values.append(val)
                    function_data += struct.pack(argfmt, *values)

                index, pointer = self._alloc(data=function_data)
                findexes.append(index)

            index, pointer = self._alloc(data=struct.pack(
                '{}I'.format(len(findexes)), *findexes))
            functions = self._List(len(findexes), index)

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

            with open(args[0], 'wb') as f:
                f.write(network + memory)
