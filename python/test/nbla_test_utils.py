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

from nnabla.utils import nnabla_pb2
from six.moves import filter
import collections
import copy
import nnabla as nn
import nnabla.ext_utils as ext_utils
import nnabla.functions as F
import nnabla.utils.converter
import numpy
import numpy as np
import pytest


def ext_to_camel(ext):
    if ext == 'cpu':
        return ''
    return ''.join([x.title() for x in ext.split('.')])


def snake_to_camel(snake):
    return ''.join([x.title() for x in snake.split('_')])


def list_context(func_name):
    try:
        import list_context_ext
        return list_context_ext.list(func_name)
    except Exception as e:
        print(e)
        return [(nn.Context(), func_name)]


def list_ctx_and_func_name(fnames):
    l = []
    for fname in fnames:
        l += [(fname, x[0], x[1]) for x in list_context(snake_to_camel(fname))]
    return l


def compute_analytical_and_numerical_grad_graph(terminal, inputs,
                                                epsilon=1e-3):
    def set_inputs(x0):
        begin = 0
        for i in inputs:
            end = begin + i.size
            i.d = x0[begin:end].reshape(i.shape)
            begin = end

    def func(x0):
        set_inputs(x0)
        terminal.forward()
        return terminal.d.copy()

    def grad(x0):
        set_inputs(x0)
        backups = [i.g.copy() for i in inputs]
        terminal.forward()
        terminal.backward()
        gx0 = []
        for i, b in zip(inputs, backups):
            gx0.append((i.g.copy() - b).flatten())
            i.g = b
        return np.concatenate(gx0)

    inputs0 = np.concatenate([i.d.flatten() for i in inputs])
    analytical_grad = grad(inputs0)
    from scipy.optimize import approx_fprime
    numerical_grad = approx_fprime(inputs0, func, epsilon)
    return analytical_grad, numerical_grad


def compute_analytical_and_numerical_grad(f, inputs, outputs, inputs0,
                                          vgrads, epsilon=1e-8, rng=None,
                                          ref_grad=None):
    """ Compute both analytical grad and numerical grad
    using given function

    f: function to test
    inputs: funcion input variable 
    outputs: function output variable
    inputs0: function inputs to calculate numerical grad
    vgrads: initianl grads of output variable
    epsilon: small value to calculate numerical grad
    rng: random number generator
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(1000))

    from scipy import optimize
    for i in inputs:
        if i is None:  # Optional argument
            continue
        i.g = rng.randn(*i.shape)
    for o, d in zip(outputs, vgrads):
        o.g = d

    def func(x0):
        bind = 0
        backups = []
        vinputs = []
        for i, i0 in zip(inputs, inputs0):
            if i is None:  # Optional argument
                continue
            vinputs += [i]
            if i0 is not None:  # Not need backward
                i.d[...] = x0[bind:bind + i.size].reshape(i.shape)
                bind += i.size
            backups.append(i.d.copy())

        f.forward(vinputs, outputs)
        for ind, i in enumerate(inputs):
            if i is None:  # Optional argument
                continue
            i.d[...] = backups[ind]
        return sum([np.sum(o.g * o.d) for o in outputs])

    def grad(x0):
        bind = 0
        backups = []
        vinputs = []
        for i, i0 in zip(inputs, inputs0):
            if i is None:  # Optional argument
                continue
            vinputs += [i]
            if i0 is not None:  # Not need backward
                i.d[...] = x0[bind:bind + i.size].reshape(i.shape)
                backups.append(i.g.copy())
                bind += i.size
            else:
                assert not i.need_grad
        f.forward(vinputs, outputs)
        f.backward(vinputs, outputs)

        bind = 0
        g = np.zeros_like(x0)
        ind = 0
        for i, i0 in zip(inputs, inputs0):
            if i is None:  # Optional argument
                continue
            if i0 is not None:  # Not need backward
                g[bind:bind + i.size] = i.g.flatten() - backups[ind].flatten()
                i.g[...] = backups[ind]
                bind += i.size
                ind += 1
        return g

    inputs0_c = np.concatenate([i0.flatten()
                                for i0 in inputs0 if i0 is not None])
    analytical_grad = grad(inputs0_c)
    numerical_grad = None
    if not ref_grad:
        numerical_grad = optimize.approx_fprime(inputs0_c, func, epsilon)
    return analytical_grad, numerical_grad


def cap_ignore_region(arr, region):
    assert len(region) == 2
    region = sorted(region)
    arr = arr.copy()
    arr[np.logical_and(arr > region[0], arr < region[0])] = region[0]
    return arr


class ArrayStats:

    def __init__(self, a):
        self.a = a
        self.mean = a.mean()
        self.std = a.std()
        self.max = a.max()
        self.min = a.min()
        self.amean = np.abs(a).mean()
        self.astd = np.abs(a).std()
        self.amax = np.abs(a).max()
        self.amin = np.abs(a).min()

    def __str__(self):
        lines = [
            'shape of {}'.format(self.a.shape),
            'raw))) mean(std): {}({}), [max, min]: [{}, {}]'.format(
                self.mean, self.std, self.max, self.min),
            'abs))) mean(std): {}({}), [max, min]: [{}, {}]'.format(
                self.amean, self.astd, self.amax, self.amin),
        ]
        return '\n'.join(lines)


class ArrayDiffStats:

    def __init__(self, a, b):
        self.astat = ArrayStats(a)
        self.bstat = ArrayStats(b)
        self.diffstat = ArrayStats(a - b)

    def __str__(self):
        lines = [
            '[diff]',
            str(self.diffstat),
            '[left]',
            str(self.astat),
            '[right]',
            str(self.bstat),
        ]
        return '\n'.join(lines)


def force_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def half_test(rng, func, finputs, hinputs, func_args, func_kwargs, backward, ctx, func_name):

    # 0. Define utility functions
    def _filter_inputs(vinputs):
        return [v for v in vinputs if v is not None]

    def _zero_grad(vs):
        for v in vs:
            if v is None:
                continue
            v.grad.zero()

    def _get_grad_copy(vinputs, backward):
        return [i.g.copy() for i, b in zip(vinputs, backward) if b and i is not None]

    def _set_output_grad_and_copy(os, grads=None):
        if grads is None:
            grads = [rng.randn(*o.shape) for o in os]
        for o, g in zip(os, grads):
            o.g = g
        return grads

    # 1. Create a float32 function.
    with nn.context_scope(ctx):
        o_f = force_tuple(func(*(finputs + func_args), **func_kwargs))
    if True in backward:
        grad_copy = _set_output_grad_and_copy(o_f)

    # 2. Get outputs of forward and backward of the float32 function.
    o_f[0].parent.forward(_filter_inputs(finputs), o_f)
    y_f = [o.d.copy() for o in o_f]
    if True in backward:
        _zero_grad(finputs)
        o_f[0].parent.backward(_filter_inputs(finputs), o_f)
        g_f = _get_grad_copy(finputs, backward)

    # 3. Create a float16 (half) function.
    ext, dtype = ctx.backend[0].split(':')
    assert dtype == 'float'
    ctx_h = ext_utils.get_extension_context(ext, type_config='half')
    with nn.context_scope(ctx_h):
        o_h = force_tuple(func(*(hinputs + func_args), **func_kwargs))
    if True in backward:
        _set_output_grad_and_copy(o_h, grad_copy)

    # 4. Get outputs of forward and backward of the float16 function.
    o_h[0].parent.forward(_filter_inputs(hinputs), o_h)
    y_h = [o.d.copy() for o in o_h]
    if True in backward:
        _zero_grad(hinputs)
        o_h[0].parent.backward(_filter_inputs(hinputs), o_h)
        g_h = _get_grad_copy(hinputs, backward)

    # 5. Check if output values are close between function data types.
    for ff, hh in zip(y_f, y_h):
        # TODO: set tol param
        assert np.allclose(ff, hh, atol=1e-1), "Checking forward half: " + str(
            ArrayDiffStats(ff, hh))
    if True not in backward:
        return
    for ff, hh in zip(g_f, g_h):
        # TODO: set tol param
        assert np.allclose(ff, hh, atol=1e-1), "Checking backward half: " + str(
            ArrayDiffStats(ff, hh))


def create_function_nnp(inputs, outputs, func_name, func_args, func_kwargs):
    if func_name is None:
        return

    for category_name, category in nnabla.utils.converter.get_category_info().items():
        if func_name in category:
            function = category[func_name]

    nnp = nnabla_pb2.NNablaProtoBuf()
    net = nnp.network.add()
    net.name = 'network1'
    net.batch_size = 1

    func = net.function.add()
    func.name = func_name
    func.type = func_name

    # Prepare input
    func_inputs = []
    data_names = []
    parameter_names = []
    input_data = []
    for n, i in enumerate(inputs):
        if i is not None:
            input_name, input_info = list(function['inputs'].items())[n]
            func_inputs.append(input_name)
            var = net.variable.add()
            var.name = input_name
            if 'parameter' in input_info and input_info['parameter']:
                parameter_names.append(input_name)

                var.type = 'Parameter'
                shape = list(i.d.shape)[:]
                if func.name == 'BatchNormalization':
                    shape = [1] + shape
                var.shape.dim.extend(shape)

                param = nnp.parameter.add()
                param.variable_name = input_name
                param.shape.dim.extend(shape)
                param.data.extend(i.d.flatten())

            else:
                input_data.append(i.d.flatten())
                data_names.append(input_name)

                var.type = 'Buffer'
                shape = list(i.d.shape)[:]
                # exclude the cases no need to extend dimension
                if input_name == 'rmean' or input_name == 't':
                    pass
                elif func.name == 'PReLU' and input_name == "x1":
                    pass
                elif func.name == 'Transpose':
                    pass
                else:
                    shape = [1] + shape
                var.shape.dim.extend(shape)

    func.input.extend(func_inputs)

    # Prepare output
    func_outputs = []
    output_data = []
    for n, o in enumerate(outputs):
        output_name = 'y{}'.format(n)
        func_outputs.append(output_name)
        var = net.variable.add()
        var.name = output_name
        var.type = 'Buffer'
        shape = list(o.d.shape)[:]
        shape = [-1] + shape
        var.shape.dim.extend(shape)
        output_data.append(o.d.flatten())

    func.output.extend(func_outputs)

    # Prepare argument
    if 'arguments' in function:
        for n, (arg_name, arg) in enumerate(function['arguments'].items()):
            param = eval('func.{}_param'.format(function['snake_name']))
            a = func_args[n]
            # This is used to fix the problem of flip (axes == None)
            if a is None:
                if 'axes' in arg_name:
                    a = len(net.variable[0].shape.dim) - 2

            if a is not None:
                if 'axis' == arg_name:
                    a += 1

                if 'axes' in arg_name:
                    if func.name == 'Transpose':
                        pass
                    else:
                        if isinstance(a, tuple) or isinstance(a, list):
                            a = list(a)
                        else:
                            a = [a]
                        a = [x + 1 for x in a]

                if isinstance(a, tuple) or isinstance(a, list):
                    if arg['type'] == 'Shape':
                        exec('param.{}.dim.extend(list(a))'.format(arg_name))
                    else:
                        exec('param.{}.extend(a)'.format(arg_name))
                elif isinstance(a, numpy.ndarray):
                    a = a.flatten()
                    if arg['type'] == 'Shape':
                        if function['snake_name'] == 'broadcast':
                            exec(
                                'param.{}.dim.extend([1] + list(a))'.format(arg_name))
                        else:
                            exec('param.{}.dim.extend(list(a))'.format(arg_name))
                    else:
                        exec('param.{}.extend(a)'.format(arg_name))
                else:
                    if 'repeated' in arg['type']:
                        exec('param.{}.extend([a])'.format(arg_name))
                    elif arg['type'] == 'string':
                        exec('param.{} = "{}"'.format(arg_name, a))
                    else:
                        if arg_name == 'base_axis':
                            a = a + 1
                        exec('param.{} = {}'.format(arg_name, a))

    # Prepare executor
    exe = nnp.executor.add()
    exe.name = 'inference'
    exe.network_name = 'network1'
    for d in data_names:
        dat = exe.data_variable.add()
        dat.variable_name = d
        dat.data_name = d

    for o in func_outputs:
        out = exe.output_variable.add()
        out.variable_name = o
        out.data_name = o

    for p in parameter_names:
        par = exe.parameter_variable.add()
        par.variable_name = p

    return nnp, input_data, output_data


def function_tester(rng, func, ref_func, inputs,
                    func_args=[], func_kwargs={},
                    atol_f=1e-6, atol_b=1e-3, atol_accum=1e-6, dstep=1e-3, backward=None,
                    ctx=None, func_name=None, ref_grad=None, disable_half_test=False):
    """ Automatic testing of forward/backward pass of `func` by comparing it
    to the reference implementation in `ref_func`.

    Syntax of `ref_func`: inputs, parametes
    Syntax of `ref_grad`: inputs, output grads, parameters
    """

    if ctx is None:
        ctx = nn.Context()
    if backward is None:
        backward = [True for _ in inputs]

    # Create Variables
    # print('create_variable')

    def create_variables(inputs, backward):
        vinputs = []
        for i, b in zip(inputs, backward):
            if i is None:
                vinputs += [None]
                continue
            vinputs += [nn.Variable(i.shape, need_grad=b)]
            vinputs[-1].data.cast(i.dtype)[...] = i
        return vinputs

    # Half test
    if not disable_half_test:
        finputs = create_variables(inputs, backward)
        hinputs = create_variables(inputs, backward)
        half_test(rng, func, finputs, hinputs, func_args,
                  func_kwargs, backward, ctx, func_name)

    vinputs = create_variables(inputs, backward)
    # Checking forward
    # print('checking forward')
    with nn.context_scope(ctx), nn.auto_forward():
        o = func(*(vinputs + func_args), **func_kwargs)
    rinputs = copy.deepcopy(inputs)  # inputs for ref_func
    refs = ref_func(*(rinputs + func_args), **func_kwargs)

    refs = force_tuple(refs)
    o = force_tuple(o)
    assert len(o) == len(refs)
    for i, ref in enumerate(refs):
        res = o[i].d
        assert np.allclose(ref, res, atol=atol_f), str(
            ArrayDiffStats(ref, res))

    # Checking function name
    try:
        import function_test_callback
        result = create_function_nnp(
            vinputs, o, func_name, func_args, func_kwargs)
        if result is not None:
            function_test_callback.callback(func_name, *result)
    except UnboundLocalError:
        pass
    except IndexError:
        pass
    except ImportError:
        pass

    # print('checking function name')
    if func_name is not None:
        assert o[0].parent.name == func_name

    # Checking backward
    # print('checking backward')
    if not True in backward:
        return

    # NNabla backward
    for v in vinputs:
        if v is None:
            continue
        if len(v.shape) == 0:
            v.g = rng.randn()
            continue
        v.g = rng.randn(*v.shape).astype(v.data.dtype)
    # Verify grad
    vinputs = create_variables(inputs, backward)
    rinputs = copy.deepcopy(inputs)
    rinputs = [rinput if test else None for rinput,
               test in zip(rinputs, backward)]
    vgrads = [rng.randn(*o_.shape) for o_ in o]

    agrads, ngrads = compute_analytical_and_numerical_grad(
        o[0].parent, vinputs, o, rinputs, vgrads, epsilon=dstep,
        rng=rng, ref_grad=ref_grad)
    if ref_grad is not None:
        rinputs = copy.deepcopy(inputs)
        doutputs = [o_.g for o_ in o]
        ngrads = ref_grad(*(rinputs + doutputs + func_args), **func_kwargs)

    assert np.allclose(ngrads, agrads, atol=atol_b), str(
        ArrayDiffStats(ngrads, agrads))

    # Check if need_grad works
    for v, b in zip(vinputs, backward):
        if not b or v is None:
            continue
        v.g = 0
        v.need_grad = False
        try:
            o[0].parent.backward(
                list(filter(lambda x: x is not None, vinputs)), o)
        except RuntimeError as e:
            continue  # TODO
        assert np.all(v.g == 0)

    # test accum=False
    for i in range(len(vinputs)):
        if vinputs[i] is None:
            continue
        v = vinputs[i]
        v.need_grad = backward[i]
    for i in range(len(vinputs)):
        if vinputs[i] is None:
            continue
        v = vinputs[i]
        if not backward[i]:
            continue
        f = o[0].parent

        # If input's grad is inplaced, the test doesn't work correctly.
        if f.inplace_grad(i):
            continue

        # Prepare function inputs
        finputs = list(filter(lambda x: x is not None, vinputs))

        # Save accum gradient result
        g = rng.randn(*v.shape)
        v.g = g
        f.forward(finputs, o)
        f.backward(finputs, o)
        true_g = v.g - g

        # Check accum=False
        accum = [j != i for j in range(len(finputs))]
        v.g = rng.randn(*v.shape)
        f.forward(finputs, o)
        f.backward(finputs, o, accum)
        assert np.allclose(
            v.g, true_g, atol=atol_accum), str(ArrayDiffStats(v.g, true_g))

        # Check accum=False with NaN gradient
        v.g = np.float32('nan')
        f.forward(finputs, o)
        f.backward(finputs, o, accum)
        assert not np.any(np.isnan(v.g))


def inplace_function_test_helper(inputs, func, func_args=[], func_kwargs={}, ctx=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(313)
    if ctx is None:
        ctx = nn.Context()
    with nn.context_scope(ctx):
        a_s = [inp * 1.0 for inp in inputs]
        y = func(*(a_s + list(func_args)), inplace=False, **func_kwargs)
        l = F.sum(y)
        a_s_i = [inp * 1.0 for inp in inputs]
        y_i = func(*(a_s_i + list(func_args)), inplace=True, **func_kwargs)
        l_i = F.sum(y_i)
    data = [(rng.randn(*inp.shape), rng.randn(*inp.shape)) for inp in inputs]
    for i in range(len(data)):
        inputs[i].d = data[i][0]
        inputs[i].g = data[i][1]
    l.forward()
    l.backward()
    grads = [inp.g.copy() for inp in inputs]
    for i in range(len(data)):
        inputs[i].d = data[i][0]
        inputs[i].g = data[i][1]
    l_i.forward()
    l_i.backward()
    grads_i = [inp.g.copy() for inp in inputs]
    for g, g_i in zip(grads, grads_i):
        assert np.allclose(g, g_i), str(ArrayDiffStats(g, g_i))


def convert_to_float2_array(x_complex, dtype=np.float32):
    real = np.real(x_complex)
    imag = np.imag(x_complex)
    real_s = real.reshape((real.shape)+(1, ))
    imag_s = imag.reshape((imag.shape)+(1, ))
    x_float2 = np.concatenate([real_s, imag_s], axis=len(real_s.shape)-1)
    return x_float2.astype(dtype)


def convert_to_complex_array(x_float2, dtype=np.complex64):
    x_real = x_float2[..., 0]
    x_imag = x_float2[..., 1]
    x_complex = x_real + 1j * x_imag
    return x_complex
