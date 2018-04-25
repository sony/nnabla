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

import nnabla.utils.nnabla_pb2 as nnabla_pb2
import nnabla.utils.converter

from .utils import create_nnabart_info


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

    def export(self, *args):
        if len(args) == 1:
            ####################################################################
            # make 2 data to save Variable Buffers in inference
            actual_buf_sizes, vidx_to_abidx = self.__save_variable_buffer()

            ####################################################################
            # Version
            version = nnabla.utils.converter.get_category_info_version()

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

                # Default function implementation is 0(float)
                function_data += struct.pack('I', 0)

                finfo = self._info._function_info[f.type]

                inputs = [vindexes_by_name[i] for i in f.input]
                index, pointer = self._alloc(data=struct.pack(
                    '{}I'.format(len(inputs)), *inputs))
                function_data += struct.pack('iI', len(inputs), index)

                outputs = [vindexes_by_name[o] for o in f.output]
                index, pointer = self._alloc(data=struct.pack(
                    '{}I'.format(len(outputs)), *outputs))
                function_data += struct.pack('iI', len(outputs), index)

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

    def __save_variable_buffer(self):
        # make the followings to save memory usage for Variable Buffer:
        #  - actual_buf_sizes(list): sizes of actual buffers, which lie unfer Variable Buffer.
        #                            indices in this list are hereinafter called 'actual buffer index'
        #  - vidx_to_abidx(dict): assignment of actual buffers to Variable Buffer.
        #                         the key and the value are Variable index and actual buffer index, respectively
        info = self._info
        buf_var_lives = self.__make_buf_var_lives(info)
        actual_buf_sizes = self.__compute_actual_buf_sizes(info, buf_var_lives)
        buf_var_refs = self.__make_buf_var_refs(info, buf_var_lives)
        vidx_to_abidx = self.__assign_actual_buf_to_variable(
            info, actual_buf_sizes, buf_var_refs)
        return (list(actual_buf_sizes), vidx_to_abidx)

    def __make_buf_var_lives(self, info):
        # buf_var_lives is to remember from when and until when each
        # Buffer Variables must be alive
        buf_var_num = len(info._variable_buffer_index)
        buf_var_lives = [_LifeSpan() for _ in range(buf_var_num)]
        name_to_vidx = {v.name: i for i,
                        v in enumerate(info._network.variable)}
        name_to_var = {v.name: v for v in info._network.variable}

        # set _LifeSpan.begin_func_idx and .end_func_idx along info._network
        for func_idx, func in enumerate(info._network.function):
            for var_name in list(func.input) + list(func.output):
                if name_to_var[var_name].type == 'Buffer':
                    var_idx = name_to_vidx[var_name]
                    buf_idx = info._buffer_ids[var_idx]
                    buf_var_life = buf_var_lives[buf_idx]
                    if buf_var_life.begin_func_idx < 0:
                        buf_var_life.begin_func_idx = func_idx
                    else:
                        # only identify a Function which first refers to the Variable
                        pass
                    buf_var_life.end_func_idx = func_idx
                else:
                    pass  # ignore 'Parameter'

        return buf_var_lives

    def __count_actual_buf(self, info, buf_var_lives):
        # count how many buffers are required at maximum based on buf_var_lives
        actual_buf_num = 0
        for func_idx, _ in enumerate(info._network.function):
            buf_num = 0
            for buf_idx, buf_var_life in enumerate(buf_var_lives):
                buf_num += int(buf_var_life.needed_at(func_idx))
            actual_buf_num = max(actual_buf_num, buf_num)
        return actual_buf_num

    def __make_buf_var_refs(self, info, buf_var_lives):
        # buf_var_refs is to store buffer indices of buffers required in each Function
        actual_buf_num = self.__count_actual_buf(info, buf_var_lives)
        shape = (len(info._network.function), actual_buf_num)
        buf_var_refs = np.empty(shape, dtype=np.int32)
        buf_var_refs[:] = -1

        # fill buf_var_refs based on buf_var_lives
        for func_idx, _ in enumerate(info._network.function):
            crsr = 0
            for buf_idx, buf_var_life in enumerate(buf_var_lives):
                if buf_var_life.needed_at(func_idx):
                    buf_var_refs[func_idx][crsr] = buf_idx
                    crsr += 1
                else:
                    pass  # only focus on buffers used in this func

        return buf_var_refs

    def __compute_actual_buf_sizes(self, info, buf_var_lives):
        # buf_size_array is to store size values of each actual buffer
        actual_buf_num = self.__count_actual_buf(info, buf_var_lives)
        buf_size_array = np.zeros(actual_buf_num, dtype=np.int32)

        # tmp_size_array is size values when only focusing on a single Function
        tmp_size_array = np.empty_like(buf_size_array, dtype=np.int32)
        for func_idx, _ in enumerate(info._network.function):
            tmp_size_array[:] = -1
            crsr = 0
            for buf_idx, buf_var_life in enumerate(buf_var_lives):
                if buf_var_life.needed_at(func_idx):
                    tmp_size_array[crsr] = info._variable_buffer_size[buf_idx]
                    crsr += 1
                else:
                    pass  # only focus on buffers used in this func

            # update sizes of actual buffers
            tmp_size_array = np.sort(tmp_size_array)
            for i in range(actual_buf_num):
                buf_size_array[i] = max(buf_size_array[i], tmp_size_array[i])

        return buf_size_array

    def __assign_actual_buf_to_variable(self, info, actual_buf_sizes, buf_var_refs):
        # create a dictionary to store assiginment of actual buffers to Variables

        # vidx_to_abidx is short for variable index to actual buffer index
        vidx_to_abidx = {}

        # actual_assigned_flags is to remember if actual buffers are assigned or not
        actual_buf_num = len(actual_buf_sizes)
        actual_assigned_flags = np.empty(actual_buf_num, dtype=np.bool)

        for func_idx, _ in enumerate(info._network.function):
            actual_assigned_flags[:] = False
            for ref_crsr in range(actual_buf_num):
                # minus buf_idx means the corresponding buffer is not needed
                buf_idx = buf_var_refs[func_idx][ref_crsr]
                if buf_idx < 0:
                    continue

                # restore assignment determined in the previous func_idx
                vidx = info._variable_buffer_index[buf_idx][0]
                if vidx in vidx_to_abidx:
                    abidx = vidx_to_abidx[vidx]
                    actual_assigned_flags[abidx] = True
                else:
                    pass  # determine assigment for this vidx in the follwoing for loop

            # determine new assigments of actual buffers to Variables
            for ref_crsr in range(actual_buf_num):
                # minus buf_idx means the corresponding buffer is not needed
                buf_idx = buf_var_refs[func_idx][ref_crsr]
                if buf_idx < 0:
                    continue

                # skip Variables to which an actual buffer is already assigned
                vidx = info._variable_buffer_index[buf_idx][0]
                if vidx in vidx_to_abidx:
                    continue

                # search for actual buffers vacant and large enough
                needed_size = info._variable_buffer_size[buf_idx]
                abidx = 0
                while abidx != actual_buf_num:
                    cond = not actual_assigned_flags[abidx]
                    cond &= needed_size <= actual_buf_sizes[abidx]
                    if cond:
                        actual_assigned_flags[abidx] = True
                        vidx_to_abidx[vidx] = abidx
                        break
                    else:
                        abidx += 1

                # increase size if buffers large enough was NOT found
                if abidx == actual_buf_num:
                    for abidx in range(actual_buf_num):
                        if not actual_assigned_flags[abidx]:
                            actual_buf_sizes[abidx] = needed_size
                            actual_assigned_flags[abidx] = True
                            vidx_to_abidx[vidx] = abidx
                            break

        return vidx_to_abidx


class _LifeSpan:
    def __init__(self):
        self.begin_func_idx = -1
        self.end_func_idx = -1

    def needed_at(self, func_idx):
        needed = self.begin_func_idx <= func_idx
        needed &= self.end_func_idx >= func_idx
        return needed
