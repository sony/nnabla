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
        return math.ceil(size/4)*4
    
    def _alloc(self, size=-1, data=b''):
        size = len(data) if size < 0 else size
        index = len(self._memory_index)
        pointer = sum(self._memory_index)
        self._memory_index.append(self._align(size))
        assert(len(data) <= size)
        self._memory_data += data
        self._memory_data += b'\n' * (self._align(len(data)) - len(data))
        return (index, pointer)
    
    def __init__(self, nnp, batch_size):
        self._info = create_nnabart_info(nnp, batch_size)

        self._List = collections.namedtuple('List', ('size', 'list_index'))
        self._Memory = collections.namedtuple('Memory', ('num_of_data', 'data_size'))

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
            version = nnabla.utils.converter.get_category_info_version()
            self._Variable = collections.namedtuple('Variable', ('id', 'shape', 'type', 'fp_pos', 'data_index'))

            variables = []
            for n, v in enumerate(self._info._network.variable):
                var = self._Variable
                var.id = n

                shape = [x if x >= 0 else self._info._batch_size for x in v.shape.dim]
                index, pointer = self._alloc(data=struct.pack('{}I'.format(len(shape)), *shape))
                var.shape = self._List(len(shape), index)

                var.type = 0 # NN_DATA_TYPE_FLOAT
                var.fp_pos = 0
                
                print(n, v.name, self._info._variable_buffer_index[n])
                if v.type == 'Parameter':
                    index, pointer = self._alloc(size=self._info._variable_sizes[self._info._variable_buffer_index[n]])
                    var.data_index = index
                elif v.type == 'Buffer':
                    var.data_index = self._info._variable_buffer_index[n] * -1

                variable = struct.pack('IiIBi',
                                       var.id,
                                       var.shape.size, var.shape.list_index,
                                       (var.type & 0xf << 4) | (var.fp_pos & 0xf),
                                       var.data_index)
                index, pointer = self._alloc(data=variable)
                variables.append(index)
            
                
            index, pointer = self._alloc(data=struct.pack('{}I'.format(len(variables)), *variables))
            variables = self._List(len(variables), index)
            functions = self._List(0, 0)
            inputs = self._List(0, 0)
            outputs = self._List(0, 0)
            memory = self._Memory(0, 0)
            
            network= struct.pack('IiIiIiIiIII',
                                 version,
                                 variables.size,
                                 variables.list_index,
                                 functions.size,
                                 functions.list_index,
                                 inputs.size,
                                 inputs.list_index,
                                 outputs.size,
                                 outputs.list_index,
                                 memory.num_of_data,
                                 memory.data_size)

            with open(args[0], 'wb') as f:
                f.write(network)
