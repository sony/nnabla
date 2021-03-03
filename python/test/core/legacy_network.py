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

from nnabla.logger import logger
import nnabla.function as F


def print_network_traceback(funcs):
    logger.critical('Network traceback:')
    for i, func in enumerate(funcs):
        logger.critical('{}{}'.format(
            '->' if i == len(funcs) - 1 else '  ', func.name))


class Network:

    def setup_function(self, func):
        try:
            func.function_instance.setup(
                func.variable_inputs, func.variable_outputs)
        except:
            logger.critical('An error occurred while setup of function {} (nn.{}) in network {}'.format(
                func.name, func.function_instance.name, self.name))
            logger.critical('Input variables:')
            for v in func.inputs:
                logger.critical('  {} (shape: {}, design_shape: {})'.format(
                    v.name, str(v.variable_instance.shape), str(v.shape)))
            logger.critical('Output variables:')
            for v in func.outputs:
                logger.critical('  {} (shape: {}, design_shape: {})'.format(
                    v.name, str(v.variable_instance.shape), str(v.shape)))
            raise
        # logger.debug('Setup: {} {}'.format(func.name, func.function_instance.name))

    def get_forward_sequence(self, loss_variables):
        forward_sequence = []
        for func in self.functions.values():
            func.forward_complete = False
        for loss in loss_variables:
            self.__forward_recursive(forward_sequence, variable=loss)
        return forward_sequence

    def __forward_recursive(self, forward_sequence, variable=None, function=None):
        if not function and variable not in self.variable_inputs:
            return
        for func in [function] if function else self.variable_inputs[variable]:
            if func.forward_complete:
                continue
            for input_function in func.input_functions:
                self.__forward_recursive(
                    forward_sequence, function=input_function)
            forward_sequence.append(func)
            func.forward_complete = True

    def forward(self, forward_sequence):
        for func in forward_sequence:
            try:
                self.forward_function(func)
            except:
                index = forward_sequence.index(func)
                print_network_traceback(
                    forward_sequence[max(0, index - 4):index + 1])
                raise

    def forward_function(self, func):
        try:
            # Uncomment when debugging expand_recurrent
            # print(func.name)
            # print(func.function_instance)
            # for n, inp in enumerate(func.variable_inputs):
            #     print('   IN:', n, inp.shape, inp.d.flatten()[0])
            func.function_instance.forward(
                func.variable_inputs, func.variable_outputs)
            # Uncomment when debugging expand_recurrent
            # for n, out in enumerate(func.variable_outputs):
            #     print('  OUT:', n, out.shape, out.d.flatten()[0])
        except:
            logger.critical('An error occurred while executing forward of function {} (nn.{}) in network {}'.format(
                func.name, func.function_instance.name, self.name))
            raise

    def get_backward_sequence(self, loss_variables, parameter_variables_and_locallr):
        class BackwardSequence:
            loss_variables = []
            variables = []
            grad_variables = []
            unused_variables = []
            parameters = []
            sequence = []
        backward_sequence = BackwardSequence()

        backward_sequence.loss_variables = [
            v.variable_instance for v in loss_variables]
        for p, lr in parameter_variables_and_locallr.items():
            if lr > 0.0:
                backward_sequence.parameters.append(p.variable_instance)

        for func in self.functions.values():
            func.backward_complete = False

        for p, local_lr in parameter_variables_and_locallr.items():
            if local_lr > 0.0:
                self.__backward_recursive(
                    backward_sequence, loss_variables, variable=p)

        for seq in backward_sequence.sequence:
            backward_sequence.variables.extend(seq.func.variable_outputs)
        for v in self.variables.values():
            vi = v.variable_instance
            if vi not in backward_sequence.variables and vi not in backward_sequence.parameters:
                backward_sequence.unused_variables.append(vi)
        return backward_sequence

    def __backward_recursive(self, backward_sequence, loss_variables, variable=None, function=None):
        # logger.debug('bwcall: {}'.format(function.name if function else ''))
        if not function and variable not in self.variable_outputs:
            # terminal variable
            return variable in loss_variables
        diff_exists = False
        for func in [function] if function else self.variable_outputs[variable]:
            if func.backward_complete:
                diff_exists = True
                continue
            func.backward_complete = True
            for output_function in func.output_functions:
                if func.output_functions:
                    diff = self.__backward_recursive(
                        backward_sequence, loss_variables, function=output_function)
                    diff_exists = diff_exists or diff
            else:
                # terminal function
                for v in loss_variables:
                    diff_exists = diff_exists or (v in func.outputs)
            if diff_exists:
                if backward_sequence is not None:
                    class BackwardSequenceItem:
                        func = None
                        accum_grad = []
                    seq = BackwardSequenceItem()
                    seq.func = func
                    for i, v in enumerate(func.variable_inputs):
                        accum = (
                            v in backward_sequence.grad_variables or v in backward_sequence.parameters)
                        seq.accum_grad.append(accum)
                        if not v in backward_sequence.grad_variables:
                            backward_sequence.grad_variables.append(v)
                    backward_sequence.sequence.append(seq)
        return diff_exists

    def prepare_backward(self, backward_sequence, parameter_zero_grad=True):
        for v in backward_sequence.unused_variables:
            v.need_grad = False
        for p in backward_sequence.parameters:
            p.need_grad = True
            if parameter_zero_grad:
                p.grad.zero()
        for v in backward_sequence.variables:
            v.need_grad = True

        # We think this should be a bug
        # for l in backward_sequence.loss_variables:
        #     l.grad.fill(1.0 / l.size)
        for l in backward_sequence.loss_variables:
            l.grad.fill(1.0)

    def backward(self, backward_sequence, parameter_zero_grad=True):
        self.prepare_backward(backward_sequence, parameter_zero_grad)
        for seq in backward_sequence.sequence:
            try:
                self.backward_function(seq)
            except:
                index = backward_sequence.sequence.index(seq)
                print_network_traceback(
                    [seq.func for seq in backward_sequence.sequence[max(0, index - 4):index + 1]])
                raise

    def backward_function(self, seq):
        try:
            seq.func.function_instance.backward(
                seq.func.variable_inputs, seq.func.variable_outputs, seq.accum_grad)
        except:
            logger.critical('An error occurred while executing backward of function {} (nn.{}) in network {}'.format(
                seq.func.name, seq.func.function_instance.name, self.name))
            raise
        # logger.debug('Backward: {} {}'.format(func.name, func.function_instance.name))

    def setup(self, optimize=False):
        if optimize:
            for func in list(self.functions.values()):
                # remove identity layer
                if func.function_instance.name[0:8] == "Identity" and not func.persistent:
                    assert(len(func.inputs) == 1)
                    assert(len(func.outputs) == 1)
                    # if the identity function is not terminal (keep terminal
                    # identity function)
                    if func.outputs[0] in self.variable_outputs:
                        next_functions = self.variable_outputs[func.outputs[0]]
                        self.variable_outputs[func.inputs[0]].remove(func)
                        self.variable_outputs[
                            func.inputs[0]].extend(next_functions)
                        for next_function in next_functions:
                            next_function.inputs = [func.inputs[0] if v == func.outputs[
                                0] else v for v in next_function.inputs]
                        del self.functions[func.name]
                        del self.variables[func.outputs[0].name]

        # create variable instances
        for variable in self.variables.values():
            if variable.variable_instance.shape != variable.shape:
                if hasattr(variable.variable_instance, 'reset_shape'):
                    variable.variable_instance.reset_shape(
                        variable.shape, force=True)
                else:
                    variable.variable_instance.reshape(
                        variable.shape, force=True)

        # setup functions
        for i, func in enumerate(self.functions.values()):
            func.variable_inputs = [v.variable_instance for v in func.inputs]
            func.variable_outputs = [v.variable_instance for v in func.outputs]
            try:
                self.setup_function(func)
            except:
                print_network_traceback(list(self.functions.values())[
                                        max(0, i - 4):i + 1])
                raise

        # set link structure to each layer
        from itertools import chain
        for func in self.functions.values():
            func.input_functions = list(chain.from_iterable(
                [self.variable_inputs[v] for v in func.inputs if v in self.variable_inputs]))
            func.output_functions = list(chain.from_iterable(
                [self.variable_outputs[v] for v in func.outputs if v in self.variable_outputs]))
            logger.debug(func.name)
            logger.debug('  in: {}'.format(
                [f.name for f in func.input_functions]))
            logger.debug(' out: {}'.format(
                [f.name for f in func.output_functions]))
