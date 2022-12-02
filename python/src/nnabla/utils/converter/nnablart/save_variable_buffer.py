# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import numpy as np


class _LifeSpan:
    def __init__(self):
        self.begin_func_idx = -1
        self.end_func_idx = -1

    def needed_at(self, func_idx):
        needed = self.begin_func_idx <= func_idx
        needed &= self.end_func_idx >= func_idx
        return needed


def __make_buf_var_lives(info):
    # buf_var_lives is to remember from when and until when each
    # Buffer Variables must be alive

    buf_var_num = len(info._variable_buffer_index)
    buf_var_lives = [_LifeSpan() for _ in range(buf_var_num)]
    name_to_vidx = {v.name: i for i,
                    v in enumerate(info._network.variable)}
    name_to_var = {v.name: v for v in info._network.variable}

    # set _LifeSpan.begin_func_idx and .end_func_idx along info._network
    final_func_idx = len(info._network.function)
    for func_idx, func in enumerate(info._network.function):
        for var_name in list(func.input) + list(func.output):
            if var_name in info._generator_variables:
                # no need to assign buffer for generator data
                pass
            if name_to_var[var_name].type == 'Buffer':
                var_idx = name_to_vidx[var_name]
                buf_idx = info._buffer_ids[var_idx]
                buf_var_life = buf_var_lives[buf_idx]
                if buf_var_life.begin_func_idx < 0:
                    if var_name in info._input_variables:
                        buf_var_life.begin_func_idx = 0
                    else:
                        buf_var_life.begin_func_idx = func_idx
                else:
                    # only identify a Function which first refers to the Variable
                    pass
                if var_name in info._output_variables:
                    buf_var_life.end_func_idx = final_func_idx
                else:
                    buf_var_life.end_func_idx = func_idx
            else:
                pass  # ignore 'Parameter'

    return buf_var_lives


def __count_actual_buf(info, buf_var_lives):
    # count how many buffers are required at maximum based on buf_var_lives
    actual_buf_num = 0
    for func_idx, _ in enumerate(info._network.function):
        buf_num = 0
        for buf_idx, buf_var_life in enumerate(buf_var_lives):
            buf_num += int(buf_var_life.needed_at(func_idx))
        actual_buf_num = max(actual_buf_num, buf_num)
    return actual_buf_num


def __make_buf_var_refs(info, buf_var_lives):
    # buf_var_refs is to store buffer indices of buffers required in each Function
    actual_buf_num = __count_actual_buf(info, buf_var_lives)
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


def __compute_actual_buf_sizes(info, buf_var_lives):
    # buf_size_array is to store size values of each actual buffer
    actual_buf_num = __count_actual_buf(info, buf_var_lives)
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


def __assign_actual_buf_to_variable(info, actual_buf_sizes, buf_var_refs):
    # create a dictionary to store assignment of actual buffers to Variables

    # vidx_to_abidx is short for variable index to actual buffer index
    vidx_to_abidx = {}

    # actual_assigned_flags is to remember if actual buffers are assigned or not
    actual_buf_num = len(actual_buf_sizes)
    actual_assigned_flags = np.empty(actual_buf_num, dtype=bool)

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
                pass  # determine assignment for this vidx in the following for loop

        # determine new assignments of actual buffers to Variables
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


def save_variable_buffer(info):
    # make the followings to save memory usage for Variable Buffer:
    #  - actual_buf_sizes(list): sizes of actual buffers, which lie under Variable Buffer.
    #                            indices in this list are hereinafter called 'actual buffer index'
    #  - vidx_to_abidx(dict): assignment of actual buffers to Variable Buffer.
    #                         the key and the value are Variable index and actual buffer index, respectively
    buf_var_lives = __make_buf_var_lives(info)
    actual_buf_sizes = __compute_actual_buf_sizes(info, buf_var_lives)
    buf_var_refs = __make_buf_var_refs(info, buf_var_lives)
    vidx_to_abidx = __assign_actual_buf_to_variable(
        info, actual_buf_sizes, buf_var_refs)

    return list(actual_buf_sizes), vidx_to_abidx
