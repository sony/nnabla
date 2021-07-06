# Copyright 2021 Sony Corporation.
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

import numpy as np


def variable_batch_size(network):
    expect_batch_size = None
    resolved_variable = []
    # All inputs of the function are parameters, so the variable to the output is special variable.
    # The special variable will not contain batch_size.
    special_variable = []

    for pf in network.forward_sequence():
        if 'base_axis' in pf.args and pf.args['base_axis'] != 0:
            base_axis = pf.args['base_axis']
            for pv_name in pf.inputs:
                if pv_name in network.variables:
                    shape = network.variables[pv_name].shape
                    expect_batch_size = np.prod(shape[:base_axis])
                    break

    if expect_batch_size is None:
        return

    # Dim 0 of shape expects to be batch size,
    # and modify the arguments of some specific functions.
    for pf in network.forward_sequence():
        normal_variable = None
        for pv_name in pf.inputs:
            if pv_name in network.variables and pv_name not in special_variable:
                normal_variable = pv_name
                break
        if normal_variable is None:
            special_variable.extend(pf.outputs)
            continue

        if 'base_axis' in pf.args and pf.args['base_axis'] != 1:
            base_axis = pf.args['base_axis']
            for pv_name in pf.inputs:
                if pv_name in network.variables and pv_name not in resolved_variable and pv_name not in special_variable:
                    shape = network.variables[pv_name].shape
                    network.variables[pv_name].shape = (np.prod(shape[:base_axis]),) + \
                        shape[base_axis:]
                    resolved_variable.append(pv_name)
            for pv_name in pf.outputs:
                if pv_name not in resolved_variable:
                    shape = network.variables[pv_name].shape
                    network.variables[pv_name].shape = (np.prod(shape[:base_axis]),) + \
                        shape[base_axis:]
                pf.args['base_axis'] = 1

        if pf.type == 'Reshape':
            arg_shape = pf.args['shape']
            assert (arg_shape[0] == expect_batch_size)
            pf.args['shape'] = [-1] + arg_shape[1:]
        elif pf.type == 'Broadcast':
            arg_shape = pf.args['shape']
            pv_name = network.variables[pf.inputs[0]]
            input_shape = [expect_batch_size] + list(pv_name.shape[1:])
            if len(input_shape) < len(arg_shape):
                diff = len(arg_shape) - len(input_shape)
                assert (np.prod(arg_shape[:diff+1]) == expect_batch_size)
                pf.args['shape'] = [-1] + arg_shape[diff+1:]
            else:
                assert (arg_shape[0] == expect_batch_size)
                pf.args['shape'] = [-1] + arg_shape[1:]
        elif pf.type in ['Constant', 'Rand', 'Randint', 'Randn', 'RandBinomial', 'RandBeta', 'RandGamma']:
            arg_shape = pf.args['shape']
            assert (arg_shape[0] == expect_batch_size)
            pf.args['shape'] = [-1] + arg_shape[1:]

    for var in network.variables.values():
        if var.shape:
            if var.name not in special_variable and var.shape[0] == expect_batch_size:
                var.shape = (-1,) + var.shape[1:]

    network.batch_size = expect_batch_size
