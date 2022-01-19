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
import re

from nnabla.utils import nnabla_pb2

from .exporter import rename_square_bracket


# Expand repeat and recurrent in nnp file.


class _TopologicalSort:
    def __init__(self, functions):
        self._orig = functions
        self._flags = {}

    def _visit(self, func):
        fname = func[0]
        if fname in self._flags:
            if self._flags[fname] == 1:
                logger.critical('Fatal error! network ins not Dag.')
                import sys
                sys.exit(-1)
            else:
                return
        else:
            if fname not in self._flags:
                self._flags[fname] = 1
            for output in func[3]:
                for f in self._orig:
                    for input in f[2]:
                        if output == input:
                            self._visit(f)

        self._flags[fname] = 2
        self._sorted.insert(0, func)

    def sorted(self):
        self._sorted = []
        for f in self._orig:
            fname = f[0]
            if fname not in self._flags:
                self._visit(f)

        return self._sorted


class NnpExpander:

    def __init__(self, nnp):
        self._nnp = nnp
        self._parameters = {}
        for param in self._nnp.parameter:
            self._parameters[param.variable_name] = True

    def _expand_repeat(self, network):

        def _search_repeat_id(mes, rid):
            return list(mes.repeat_id).index(rid) if rid in mes.repeat_id else None

        def _add_suffix(name, suffix, num):
            return '{}_{}_{}'.format(name, suffix, num)

        ########################################################################
        # Prepare output network message
        net = nnabla_pb2.NNablaProtoBuf().network.add()
        net.CopyFrom(network)

        ########################################################################
        # Finish when repeat_info is not present
        if len(net.repeat_info) == 0:
            return net

        ########################################################################
        # Use first repeat_info
        ri = net.repeat_info[0]
        del net.repeat_info[0]

        ########################################################################
        # Expand variables
        net.ClearField('variable')
        for vpos, var in enumerate(network.variable):
            if var.type == 'Parameter':
                if var.name not in self._parameter_original_names:
                    self._parameter_original_names[var.name] = []

            pos = _search_repeat_id(var, ri.id)
            if pos is not None:
                for i in range(ri.times):

                    if var.type == 'Parameter':
                        if self._old_version_param_name:
                            name = _add_suffix(var.name, ri.id, i)
                        else:
                            name = var.name.replace(
                                '{{{}}}'.format(ri.id), '_{}'.format(i))
                        self._parameter_original_names[var.name].append(name)
                    else:
                        name = _add_suffix(var.name, ri.id, i)

                    v = net.variable.add()
                    v.CopyFrom(var)
                    v.name = name
                    del v.repeat_id[pos]
            else:
                if var.type == 'Parameter' and len(var.repeat_id) == 0 and len(self._parameter_original_names[var.name]) == 0:
                    self._parameter_original_names[var.name].append(var.name)
                v = net.variable.add()
                v.CopyFrom(var)

        ########################################################################
        # Expand functions
        ########################################################################

        ########################################################################
        # Prepare delayed inputs
        delay_var = {}
        for fpos, func in enumerate(network.function):
            if func.type == 'Delay':
                if func.recurrent_param.repeat_id == ri.id:
                    delay_var[func.output[0]] = []
                    for i in range(ri.times):
                        if i == 0:
                            delay_var[func.output[0]].append(func.input[1])
                        else:
                            v = func.input[0]
                            v = _add_suffix(v, ri.id, i-1)
                            delay_var[func.output[0]].append(v)

        ########################################################################
        # Prepare repeat end inputs
        repeat_end_var = {}
        for fpos, func in enumerate(network.function):
            if func.type == 'RepeatEnd':
                if func.repeat_param.repeat_id == ri.id:
                    repeat_end_var[func.output[0]] = []
                    for i in range(func.repeat_param.times):
                        repeat_end_var[func.output[0]].append(_add_suffix(
                            func.input[0], func.repeat_param.repeat_id, i))

        ########################################################################
        # Prepare repeat start inputs
        repeat_start_var = {}
        for fpos, func in enumerate(network.function):
            if func.type == 'RepeatStart':
                if func.repeat_param.repeat_id == ri.id:
                    repeat_start_var[func.output[0]] = []
                    for i in range(ri.times):
                        if i == 0:
                            v = func.input[0]
                            if v in repeat_end_var:
                                v = repeat_end_var[v][ri.times-1]
                            repeat_start_var[func.output[0]].append(v)
                        else:
                            v = func.input[1]
                            if v in repeat_end_var:
                                v = repeat_end_var[v][i-1]
                            else:
                                v = _add_suffix(v, ri.id, i-1)
                            repeat_start_var[func.output[0]].append(v)

        ########################################################################
        # Expand network
        net.ClearField('function')
        for fpos, func in enumerate(network.function):
            if func.type == 'RepeatStart' or func.type == 'RepeatEnd':
                if func.repeat_param.repeat_id == ri.id:
                    continue
            if func.type == 'Delay':
                if func.recurrent_param.repeat_id == ri.id:
                    continue
            if func.type == 'RecurrentInput':
                if func.recurrent_param.repeat_id == ri.id:

                    f = net.function.add()
                    f.CopyFrom(func)
                    f.type = 'Split'
                    f.split_param.axis = func.recurrent_param.axis

                    f.ClearField('output')
                    for i in range(ri.times):
                        f.output.append(_add_suffix(func.output[0], ri.id, i))

                    pos = _search_repeat_id(func, ri.id)
                    del f.repeat_id[pos]
                    f.ClearField('recurrent_param')
                    continue

            if func.type == 'RecurrentOutput':
                if func.recurrent_param.repeat_id == ri.id:
                    f = net.function.add()
                    f.CopyFrom(func)
                    f.type = 'Stack'
                    f.stack_param.axis = func.recurrent_param.axis

                    f.ClearField('input')
                    for i in range(ri.times):
                        f.input.append(_add_suffix(func.input[0], ri.id, i))

                    f.ClearField('recurrent_param')
                    continue

            pos = _search_repeat_id(func, ri.id)
            if pos is not None:

                for i in range(ri.times):

                    f = net.function.add()
                    f.CopyFrom(func)

                    del f.repeat_id[pos]

                    f.name = _add_suffix(func.name, ri.id, i)
                    for n, v in enumerate(func.input):
                        vname = None
                        if v in self._parameter_original_names:
                            if len(self._parameter_original_names[v]) == ri.times:
                                vname = self._parameter_original_names[v][i]
                            else:
                                vname = v
                        elif v in repeat_start_var:
                            vname = repeat_start_var[v][i]
                        elif v in repeat_end_var:
                            vname = repeat_end_var[v][i]
                        elif v in delay_var:
                            vname = delay_var[v][i]
                        else:
                            vname = _add_suffix(v, ri.id, i)
                        f.input[n] = vname
                    for n, v in enumerate(func.output):
                        vname = _add_suffix(v, ri.id, i)
                        f.output[n] = vname

            else:
                f = net.function.add()
                f.CopyFrom(func)
                for n, v in enumerate(func.input):
                    if v in repeat_end_var:
                        vname = repeat_end_var[v][ri.times-1]
                        f.input[n] = vname

        return self._expand_repeat(net)

    def _expand_network(self, network):
        self._parameter_original_names = collections.OrderedDict()

        print(' Expanding {}.'.format(network.name))

        repeat_ids = collections.OrderedDict()
        for ri in network.repeat_info:
            repeat_ids[ri.id] = ri.times

        # Check whether parameter name complies with old rule.
        self._old_version_param_name = False
        for param in self._parameters:
            for ri in repeat_ids:
                m = re.search('{}\[([0-9]+)\]$'.format(ri), param)
                if m:
                    if int(m.group(1)) < repeat_ids[ri]:
                        self._old_version_param_name = True

        # Expand repeat
        network = self._expand_repeat(network)

        functions = []
        for func in network.function:
            functions.append((func.name,
                              func.type,
                              [n for n in func.input],
                              [n for n in func.output]))

        sorted_functions = self._sort_functions(functions)
        func_list = []
        for f in functions:
            func_list.append(f[0])

        net = nnabla_pb2.NNablaProtoBuf().network.add()
        net.CopyFrom(network)
        net.ClearField('function')
        for f in sorted_functions:
            func = net.function.add()
            func.CopyFrom(network.function[func_list.index(f[0])])

        return net

    def _sort_functions(self, orig_functions):
        return _TopologicalSort(orig_functions).sorted()

    def _expand_parameter_variable(self, proto):
        names = []

        for pv in proto.parameter_variable:
            if pv.variable_name in self._parameter_original_names:
                for n in self._parameter_original_names[pv.variable_name]:
                    names.append(n)
            else:
                names.append(pv.variable_name)

        proto.ClearField('parameter_variable')
        for n in sorted(names):
            pv = proto.parameter_variable.add()
            pv.variable_name = n

    def execute(self):
        nnp = nnabla_pb2.NNablaProtoBuf()
        nnp.CopyFrom(self._nnp)

        nnp.ClearField('network')
        for network in self._nnp.network:
            net = nnp.network.add()
            net.CopyFrom(self._expand_network(network))

        for optimizer in nnp.optimizer:
            self._expand_parameter_variable(optimizer)

        for executor in nnp.executor:
            self._expand_parameter_variable(executor)

        rename_square_bracket(nnp)

        return nnp
