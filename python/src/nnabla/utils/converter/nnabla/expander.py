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

from nnabla.utils import nnabla_pb2

# Expand repeat and recurrent in nnp file.


class TopologicalSort:
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

    def sort_functions(self, orig_functions):
        return TopologicalSort(orig_functions).sorted()

    def expanded_suffixes(self, repeat_ids, index=0, suffix=''):
        suffixes = []
        rid = repeat_ids[index]
        rparam = self.repeat_params[rid]
        for n in range(rparam['length']):
            if index < len(repeat_ids)-1:
                suffixes += self.expanded_suffixes(
                    repeat_ids, index+1, ('{}_{}[{}]'.format(suffix, rid, n)))
            else:
                suffixes.append('{}_{}[{}]'.format(suffix, rid, n))
        return suffixes

    def expand_network(self, network):
        print(' Expanding {}.'.format(network.name))

        # Get repeat info
        self.repeat_params = {}
        for func in network.function:

            if func.HasField('repeat_param'):
                if func.repeat_param.repeat_id != '' and \
                   func.repeat_param.times > 0:
                    self.repeat_params[func.repeat_param.repeat_id] = \
                        {'length': func.repeat_param.times}

            if func.HasField('recurrent_param'):
                if func.recurrent_param.repeat_id != '' and \
                        func.recurrent_param.length > 0 and \
                        func.recurrent_param.axis > 0:
                    self.repeat_params[func.recurrent_param.repeat_id] = \
                        {'length': func.recurrent_param.length,
                         'axis': func.recurrent_param.axis}

        # Expand variables
        self.var_proto_by_name = collections.OrderedDict()
        for v in network.variable:
            self.var_proto_by_name[v.name] = v

        self.variable_by_name = collections.OrderedDict()
        for variable in network.variable:
            if len(variable.repeat_id) > 0:
                self.variable_by_name[variable.name] = []
                for suffix in self.expanded_suffixes(variable.repeat_id):
                    self.variable_by_name[variable.name].append(
                        variable.name + suffix)
            else:
                self.variable_by_name[variable.name] = [variable.name]

        network.ClearField('variable')
        for vname in self.variable_by_name:
            for vname_expanded in self.variable_by_name[vname]:
                proto = network.variable.add()
                proto.CopyFrom(self.var_proto_by_name[vname])
                proto.name = vname_expanded
                proto.ClearField('repeat_id')

        # Expand functions
        self.func_proto_by_name = collections.OrderedDict()
        for f in network.function:
            self.func_proto_by_name[f.name] = f

        delayed_inputs = {}
        repeatstart_inputs = {}
        repeatend_inputs = {}
        for func in network.function:
            if func.type == 'Delay':
                delayed_inputs[func.output[0]] = func.input

            if func.type == 'RepeatStart':
                repeatstart_inputs[func.output[0]] = func.input

            if func.type == 'RepeatEnd':
                suffix = '_{}[{}]'.format(
                    func.repeat_param.repeat_id, func.repeat_param.times-1)
                repeatend_inputs[func.output[0]] = func.input[0] + suffix

        self.func_proto_by_expanded_name = collections.OrderedDict()
        functions = []
        for func in network.function:
            if func.type == 'RecurrentInput':
                fname = 'Split_' + func.repeat_id[0]
                ftype = 'Split'
                finput = [n for n in func.input]
                foutput = []
                for suffix in self.expanded_suffixes(func.repeat_id):
                    foutput.append(func.output[0] + suffix)
                functions.append((fname, ftype, finput, foutput,
                                  self.repeat_params[func.recurrent_param.repeat_id]))
            elif func.type == 'RecurrentOutput':
                fname = 'Stack_' + func.recurrent_param.repeat_id
                ftype = 'Stack'
                finput = []
                for suffix in self.expanded_suffixes([func.recurrent_param.repeat_id]):
                    finput.append(func.input[0] + suffix)
                foutput = [n for n in func.output]
                functions.append((fname, ftype, finput, foutput,
                                  self.repeat_params[func.recurrent_param.repeat_id]))
            elif func.type == 'Delay' or func.type == 'RepeatStart' or func.type == 'RepeatEnd':
                pass
            else:
                if len(func.repeat_id) > 0:
                    prev_suffix = ''
                    for i, suffix in enumerate(self.expanded_suffixes(func.repeat_id)):
                        fname = func.name + suffix
                        self.func_proto_by_expanded_name[fname] = self.func_proto_by_name[func.name]
                        finput = []

                        for n in func.input:
                            if n in delayed_inputs:
                                if i == 0:
                                    finput.append(delayed_inputs[n][1])
                                else:
                                    finput.append(
                                        delayed_inputs[n][0] + prev_suffix)
                            elif n in repeatstart_inputs:
                                if i == 0:
                                    finput.append(repeatstart_inputs[n][0])
                                else:
                                    finput.append(
                                        repeatstart_inputs[n][1] + prev_suffix)
                            else:
                                nn = n + suffix
                                if nn in self.variable_by_name[n]:
                                    finput.append(nn)
                                else:
                                    finput.append(n)

                        foutput = [n + suffix for n in func.output]
                        functions.append(
                            (fname, func.type, finput, foutput, None))
                        prev_suffix = suffix

                else:
                    self.func_proto_by_expanded_name[func.name] = self.func_proto_by_name[func.name]

                    finput = []
                    for n in func.input:
                        if n in repeatend_inputs:
                            finput.append(repeatend_inputs[n])
                        else:
                            finput.append(n)

                    functions.append((func.name,
                                      func.type,
                                      finput,
                                      [n for n in func.output],
                                      None))

        # for n, (fname, ftype, finput, foutput, repeat_param) in enumerate(functions):
        #     print('orig', fname)
        functions = self.sort_functions(functions)
        # for n, (fname, ftype, finput, foutput, repeat_param) in enumerate(functions):
        #     print('sorted', fname)

        network.ClearField('repeat_info')
        network.ClearField('function')

        for func in functions:
            fname, ftype, finput, foutput, repeat_param = func
            proto = network.function.add()
            if ftype == 'Split':
                proto.split_param.axis = repeat_param['axis']
            elif ftype == 'Stack':
                proto.stack_param.axis = repeat_param['axis']
            else:
                proto.CopyFrom(self.func_proto_by_expanded_name[fname])
                proto.ClearField('repeat_id')
                proto.ClearField('input')
                proto.ClearField('output')

            proto.name = fname
            proto.type = ftype
            for i in finput:
                proto.input.append(i)
            for o in foutput:
                proto.output.append(o)

    def expand_parameter_variable(self, proto):
        names = []
        for pv in proto.parameter_variable:
            if pv.variable_name in self.variable_by_name:
                for n in self.variable_by_name[pv.variable_name]:
                    names.append(n)
            else:
                names.append(pv.variable_name)
        proto.ClearField('parameter_variable')
        for n in sorted(names):
            pv = proto.parameter_variable.add()
            pv.variable_name = n

    def expand(self):
        nnp = nnabla_pb2.NNablaProtoBuf()
        nnp.CopyFrom(self._nnp)
        for network in nnp.network:
            self.expand_network(network)
        for optimizer in nnp.optimizer:
            self.expand_parameter_variable(optimizer)
        for executor in nnp.executor:
            self.expand_parameter_variable(executor)

        return nnp
