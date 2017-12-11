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


class NnpExpander:
    def __init__(self, nnp):
        self._nnp = nnp

    def expand_recurrent(self, orig_functions, recurrent_input=[], recurrent_param=None):
        if recurrent_param is not None:
            functions0 = []

            # Split recurrent length
            fname_base = 'Split_{}'.format(recurrent_param.repeat_id)
            fname = fname_base

            count = 2
            while fname in self.func_proto_by_name:
                fname = '{}_{}'.format(fname_base, count)
                count += 1

            foutput = []
            for i in range(recurrent_param.length):
                o0, o1 = orig_functions[0][4][0]
                foutput.append(
                    ('{}_{}[{}]'.format(o0, recurrent_param.repeat_id, i), o1))

            functions0.append(
                (fname, '', 'Split', recurrent_input, foutput, recurrent_param))

            delay_inputs = {}
            for n, func in enumerate(orig_functions):
                fname, fbasename, ftype, finput, foutput, rp = func
                if ftype == 'Delay':
                    delay_inputs[foutput[0][0]] = finput

            outputs = []
            for i in range(recurrent_param.length):
                suffix = '_{}[{}]'.format(recurrent_param.repeat_id, i)
                for n, func in enumerate(orig_functions):
                    fname, fbasename, ftype, finput, foutput, rp = func
                    expand_fname = fname + suffix

                    if ftype != 'Delay':
                        expand_input = []
                        for i0, i1 in finput:
                            if i0 in delay_inputs:
                                if i == 0:
                                    expand_input.append(delay_inputs[i0][1])
                                else:
                                    expand_input.append(('{}_{}[{}]'.format(
                                        delay_inputs[i0][0][0], recurrent_param.repeat_id, i - 1), delay_inputs[i0][0][1]))
                            else:
                                expand_input.append((i0 + suffix, i1))

                        expand_output = []
                        for i0, i1 in foutput:
                            expand_output = []
                            if i0 in delay_inputs:
                                expand_input.append(
                                    ('{}_{}'.format(delay_inputs[i0][0][0], suffix), delay_inputs[i0][0][1]))
                            else:
                                expand_output.append((i0 + suffix, i1))

                        for x1, x2 in (expand_input + expand_output):
                            if x2 not in self.variable_by_name:
                                self.variable_by_name[x2] = []
                            if x1 not in self.variable_by_name[x2]:
                                self.variable_by_name[x2].append(x1)
                            if x1 not in self.variable_by_expanded_name:
                                self.variable_by_expanded_name[x1] = x2
                        functions0.append(
                            (expand_fname, fbasename, ftype, expand_input, expand_output, rp))

                    if n == len(orig_functions) - 1:
                        outputs = outputs + expand_output

            # Stack recurrent outputs
            fname_base = 'Stack_{}'.format(recurrent_param.repeat_id)
            fname = fname_base

            count = 2
            while fname in self.func_proto_by_name:
                fname = '{}_{}'.format(fname_base, count)
                count += 1

            vname_base = 'Stack_{}'.format(recurrent_param.repeat_id)
            vname = vname_base

            count = 2
            while vname in self.var_proto_by_name:
                vname = '{}_{}'.format(vname_base, count)
                count += 1

            foutput = [(vname, vname)]
            functions0.append(
                (fname, '', 'Stack', outputs, foutput, recurrent_param))

            orig_functions = functions0

        functions = []

        recurrent_output = None
        current_recurrent_id = ''
        in_recurrent = False

        for func in orig_functions:
            fname, fbasename, ftype, finput, foutput, recurrent_param = func
            if ftype == 'RecurrentInput':
                recurrent_length = self.var_proto_by_name[finput[0]
                                                          [1]].shape.dim[recurrent_param.axis]
                recurrent_funcs = []
                recurrent_input = finput
                in_recurrent = True
                current_recurrent_id = recurrent_param.repeat_id
            elif ftype == 'RecurrentOutput' and in_recurrent and recurrent_param.repeat_id == current_recurrent_id:
                assert recurrent_length == recurrent_param.length
                functions += self.expand_recurrent(
                    recurrent_funcs, recurrent_input, recurrent_param)
                recurrent_output = functions[-1][4]
                in_recurrent = False
            else:
                if in_recurrent:
                    recurrent_funcs.append(func)
                else:
                    if recurrent_output is not None:
                        finput[0] = recurrent_output[0]
                        functions.append(
                            (fname, fbasename, ftype, finput, foutput, recurrent_param))
                        recurrent_output = None
                    else:
                        functions.append(func)

        return functions

    def expand_loop(self, orig_functions, inputs={}, repeat_id='', times=0):
        if times > 0:
            functions0 = []
            for i in range(times):
                suffix = '_{}[{}]'.format(repeat_id, i)
                for n, func in enumerate(orig_functions):
                    fname, fbasename, ftype, finput, foutput, repeat_param = func
                    expand_input = [(x1 + suffix, x2) for x1, x2 in finput]
                    expand_output = [(x1 + suffix, x2) for x1, x2 in foutput]
                    expand_fname = fname + suffix
                    if n == 0:
                        if i == 0:
                            i0 = inputs[finput[0][0]][0]
                            if i0 in self.variable_by_expanded_name:
                                i1 = self.variable_by_expanded_name[i0]
                            else:
                                i1 = i0
                            expand_input[0] = (i0, i1)
                        else:
                            i0 = inputs[finput[0][0]][1] + \
                                '_{}[{}]'.format(repeat_id, i - 1)
                            i1 = self.variable_by_expanded_name[i0]
                            expand_input = [(i0, i1)]

                    for x1, x2 in (expand_input + expand_output):
                        if x2 not in self.variable_by_name:
                            self.variable_by_name[x2] = []
                        if x1 not in self.variable_by_name[x2]:
                            self.variable_by_name[x2].append(x1)
                        if x1 not in self.variable_by_expanded_name:
                            self.variable_by_expanded_name[x1] = x2

                    functions0.append(
                        (expand_fname, fbasename, ftype, expand_input, expand_output, repeat_param))
            orig_functions = functions0

        functions = []
        loop_output = None
        in_loop = False
        current_repeat_id = ''

        for func in orig_functions:
            fname, fbasename, ftype, finput, foutput, repeat_param = func
            if ftype == 'RepeatStart' and not in_loop:
                loop_funcs = []
                loop_inputs = {}
                in_loop = True
                current_repeat_id = repeat_param.repeat_id
                for i1, i2 in finput:
                    o1, o2 = foutput[0]
                    if o1 not in loop_inputs:
                        loop_inputs[o1] = []
                    loop_inputs[o1].append(i1)
            elif ftype == 'RepeatEnd' and in_loop and repeat_param.repeat_id == current_repeat_id:
                functions += self.expand_loop(loop_funcs, loop_inputs,
                                              current_repeat_id, repeat_param.times)
                if len(functions) > 0:
                    loop_output = functions[-1][4]
                in_loop = False
            else:
                if in_loop:
                    loop_funcs.append(func)
                else:

                    if loop_output is not None:
                        finput_new = []
                        for n, inp in enumerate(finput):
                            if n < len(loop_output):
                                finput_new.append(loop_output[n])
                            else:
                                finput_new.append(inp)
                        functions.append(
                            (fname, fbasename, ftype, finput_new, foutput, repeat_param))
                        loop_output = None
                    else:
                        functions.append(func)

        return functions

    # Simple topological sort.
    def sort_functions(self, orig_functions):

        all_inputs = []
        for y in [x[3] for x in orig_functions]:
            for z in y:
                all_inputs.append(z[1])
        all_inputs = set(all_inputs)
        all_outputs = []
        for y in [x[4] for x in orig_functions]:
            for z in y:
                all_outputs.append(z[1])
        all_outputs = set(all_outputs)

        variables_for_sorting = all_inputs & all_outputs

        # remove repeatstart second input from sort variable
        for di in [x[3][1][1] for x in filter(lambda n: n[2] == 'RepeatStart', [f for f in orig_functions])]:
            variables_for_sorting = variables_for_sorting - {di}

        # remove delay input/output from sort variable
        for di, do in [[x[3][1][1], x[4][0][1]] for x in filter(lambda n: n[2] == 'Delay', [f for f in orig_functions])]:
            variables_for_sorting = variables_for_sorting - {di, do}

        prev = orig_functions.pop(0)

        output_wait_list = []
        for o0, o1 in prev[4]:
            output_wait_list.append(o1)
        output_wait_list = set(output_wait_list) & variables_for_sorting

        functions = [prev]
        founds = []

        while(len(orig_functions) > 0):
            found_list = set()
            for n, func in enumerate(orig_functions):
                input_list = set([f[1] for f in func[3]]
                                 ) & variables_for_sorting
                if input_list <= output_wait_list:  # input_list is subset of output_wait_list
                    found_list = found_list | input_list
                    founds.append(n)

            #output_wait_list = output_wait_list - found_list

            for found in sorted(founds)[::-1]:
                prev = orig_functions.pop(found)
                functions.append(prev)
                wait_list = []
                for o0, o1 in prev[4]:
                    wait_list.append(o1)
                output_wait_list = (output_wait_list | set(
                    wait_list)) & variables_for_sorting
            founds = []

        return functions

    def expand_network(self, network):
        print(' Expanding {}.'.format(network.name))

        repeat_end_outputs = {}
        for fend in filter(lambda n: n.type == 'RepeatEnd', list(network.function)):
            for i, o in zip(list(fend.input), list(fend.output)):
                repeat_end_outputs[o] = i

        self.var_proto_by_name = collections.OrderedDict()
        for v in network.variable:
            self.var_proto_by_name[v.name] = v
        self.func_proto_by_name = collections.OrderedDict()
        for f in network.function:
            self.func_proto_by_name[f.name] = f

        # Before expanding loop network must be topologically sorted.
        functions = self.sort_functions([(func.name,
                                          func.name,
                                          func.type,
                                          [(n1, n2) for n1, n2 in zip(
                                              list(func.input), list(func.input))],
                                          [(n1, n2) for n1, n2 in zip(
                                              list(func.output), list(func.output))],
                                          func.repeat_param if func.HasField('repeat_param')
                                          else func.recurrent_param)
                                         for func in network.function])

        # Expand repeat.
        self.variable_by_name = collections.OrderedDict()
        self.variable_by_expanded_name = collections.OrderedDict()
        functions = self.expand_loop(functions)

        # Expand recuurent network.
        functions = self.expand_recurrent(functions)

        output_variables = []
        all_suffixes = []
        for fname, fbasename, ftype, finput, foutput, repeat_param in functions:
            all_suffixes.append(fname[len(fbasename):])
            for o1, o2 in foutput:
                output_variables.append(o1)

        network.ClearField('repeat_info')
        network.ClearField('variable')
        network.ClearField('function')

        for v, vbase in self.variable_by_expanded_name.items():
            proto = network.variable.add()
            proto.CopyFrom(self.var_proto_by_name[vbase])
            proto.name = v
            proto.ClearField('repeat_id')

        for func in functions:
            fname, fbasename, ftype, finput, foutput, repeat_param = func
            proto = network.function.add()

            if fbasename == '':
                if ftype == 'Split':
                    proto.split_param.axis = repeat_param.axis
                elif ftype == 'Stack':
                    proto.stack_param.axis = repeat_param.axis
            else:
                proto.CopyFrom(self.func_proto_by_name[fbasename])

            proto.ClearField('repeat_id')
            proto.ClearField('input')
            proto.ClearField('output')
            for i0, i1 in finput:
                proto.input.append(i0)
            for o0, o1 in foutput:
                proto.output.append(o0)
            proto.name = fname
            proto.type = ftype

    def expand(self):
        nnp = nnabla_pb2.NNablaProtoBuf()
        nnp.CopyFrom(self._nnp)
        for network in nnp.network:
            self.expand_network(network)
        return nnp
