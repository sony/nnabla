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
import copy
import nnabla as nn
import nnabla.ext_utils as ext_utils
import nnabla.functions as F
import nnabla.utils.converter
from nnabla.testing import assert_allclose
import numpy
import numpy as np
from numpy.core import function_base


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


def list_ctx_and_func_name2(fnames):
    l = []
    for fname, func_name in fnames:
        l += [(fname, x[0], x[1]) for x in list_context(func_name)]
    return l


def randn(rng, *shape):
    return np.asarray(rng.randn(*shape), dtype=np.float32)


def compute_analytical_and_numerical_grad_graph(terminal, inputs,
                                                epsilon=1e-3,
                                                recompute_graph=True):
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
        if recompute_graph:
            terminal.forward()
            terminal.backward()
        gx0 = []
        for i, b in zip(inputs, backups):
            if recompute_graph:
                gx0.append((i.g.copy() - b).flatten())
            else:
                gx0.append(i.g.copy().flatten())
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
    inputs: function input variable 
    outputs: function output variable
    inputs0: function inputs to calculate numerical grad
    vgrads: initial grads of output variable
    epsilon: small value to calculate numerical grad
    rng: random number generator
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(1000))

    from scipy import optimize
    for i in inputs:
        if i is None:  # Optional argument
            continue
        i.g = randn(rng, *i.shape)
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
    arr0 = arr.copy()
    arr[np.logical_and(arr > region[0], arr < region[1])] = region[0]
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
            '',
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


def force_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def half_test(rng, func, finputs, hinputs, func_args, func_kwargs, backward, ctx, func_name, atol=1e-1):

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
            grads = [randn(rng, *o.shape) for o in os]
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
    ctx_h.device_id = ctx.device_id
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
        assert_allclose(ff, hh, atol=atol)
    if True not in backward:
        return
    for ff, hh in zip(g_f, g_h):
        # TODO: set tol param
        assert_allclose(ff, hh, atol=atol)


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
            if len(list(function['inputs'].items())) == 1:
                input_name, input_info = list(function['inputs'].items())[0]
                if 'variadic' in input_info and input_info['variadic']:
                    input_name += str(n)
            else:
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
                elif func.name == 'Concatenate':
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
            if not func_args and not func_kwargs:
                continue
            if func.name == 'Interpolate':
                del func_args[0]
            if n < len(func_args):
                a = func_args[n]
            else:
                if func.name == 'Concatenate' or func.name == 'Stack':
                    a = func_kwargs['axis']
                else:
                    a = func_kwargs.get('keepdims')
            # This is used to fix the problem of flip (axes == None)
            if a is None:
                f = ['Sum', 'Mean', 'Max', 'Min', 'Prod']
                if 'axes' in arg_name:
                    if func.name in f:
                        a = net.variable[0].shape.dim[:-1]
                        a = [x - 1 for x in a]
                    else:
                        a = len(net.variable[0].shape.dim) - 2

            if a is not None:
                if 'axis' == arg_name:
                    if func.name == 'Concatenate':
                        pass
                    else:
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
                    ctx=None, func_name=None, ref_grad=None, disable_half_test=False, atol_half=1e-1):
    """ Automatic testing of forward/backward pass of `func` by comparing it
    to the reference implementation in `ref_func`.

    Syntax of `ref_func`: inputs, parameters
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
                  func_kwargs, backward, ctx, func_name, atol=atol_half)

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
        assert_allclose(ref, res, atol=atol_f)

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
            v.g = randn(rng)
            continue
        v.g = randn(rng, *v.shape)
    # Verify grad
    vinputs = create_variables(inputs, backward)
    rinputs = copy.deepcopy(inputs)
    rinputs = [rinput if test else None for rinput,
               test in zip(rinputs, backward)]
    vgrads = [randn(rng, *o_.shape) for o_ in o]

    agrads, ngrads = compute_analytical_and_numerical_grad(
        o[0].parent, vinputs, o, rinputs, vgrads, epsilon=dstep,
        rng=rng, ref_grad=ref_grad)
    if ref_grad is not None:
        rinputs = copy.deepcopy(inputs)
        doutputs = [o_.g for o_ in o]
        ngrads = ref_grad(*(rinputs + doutputs + func_args), **func_kwargs)

    assert_allclose(ngrads, agrads, atol=atol_b)

    # Check if need_grad works
    for v, b in zip(vinputs, backward):
        if not b or v is None:
            continue
        v.grad.zero()
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
        g = randn(rng, *v.shape)
        v.g = g
        f.forward(finputs, o)
        f.backward(finputs, o)
        true_g = v.g - g

        # Check accum=False
        accum = [j != i for j, vv in enumerate(vinputs) if vv is not None]
        v.g = randn(rng, *v.shape)
        f.forward(finputs, o)
        f.backward(finputs, o, accum)
        assert_allclose(
            v.g, true_g, atol=atol_accum)

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
    data = [(randn(rng, *inp.shape), randn(rng, *inp.shape)) for inp in inputs]
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
        assert_allclose(g, g_i)


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


def backward_function_tester(rng, func, ref_func, inputs,
                             func_args=[], func_kwargs={},
                             atol_f=1e-6, atol_b=1e-3, atol_accum=1e-3, dstep=1e-3, backward=None,
                             ctx=None, func_name=None, ref_grad=None, disable_half_test=False, atol_half=1e-1):
    """Backward function tester

    In the forward test, it compares the results of nn.grad and `func`.backward.
    In the backward test, it compares the analytical gradients and numerical gradient with `grad_outputs`.
    """
    # TODO: half

    from scipy.optimize import approx_fprime

    if ctx is None:
        ctx = nn.Context()
    if backward is None:
        backward = [True if i is not None else False for i in inputs]

    # TODO: Remove set_default_context after adding ctx to BackwardFunction.
    nn.set_default_context(ctx)

    # Create Variables
    def create_variables(inputs, backward):
        vinputs = []
        for i, b in zip(inputs, backward):
            if i is None:
                vinputs += [None]
                continue
            vinputs += [nn.Variable(i.shape, need_grad=b)]
            vinputs[-1].data.cast(i.dtype)[...] = i
        return vinputs

    # Create grad_outputs
    def create_grad_outputs(outputs):
        grad_outputs = []
        for o in outputs:
            if o.shape == ():
                go = nn.NdArray.from_numpy_array(np.array(randn(rng)))
                #go = nn.NdArray.from_numpy_array(np.array(1.0))
            else:
                go = nn.NdArray.from_numpy_array(randn(rng, *o.shape))
                #go = nn.NdArray.from_numpy_array(np.ones(o.shape))

            grad_outputs.append(go)
        return grad_outputs

    # Fill grads
    def fill_grads(vinputs, grads):
        for vi, gd in zip(vinputs, grads):
            if vi is None:
                continue
            vi.g = gd

    # Fill grads
    def zero_grads(vinputs):
        for vi in vinputs:
            if vi is None:
                continue
            vi.grad.zero()
        return

    # Gradient penalty on grads
    def gradient_penalty2(grads):
        gp2 = 0.0
        for g in grads:
            gp2 += F.sum(g ** 2.0)
        return gp2
    # Product sum

    def prod_sum(inputs0, inputs1):
        out = 0.0
        for inp0, inp1 in zip(inputs0, inputs1):
            out += inp0 * nn.Variable(inp1.shape).apply(data=inp1)
        return out
    # Set inputs for the numerical gradients

    def set_inputs(inputs0, vinputs):
        begin = 0
        for i in vinputs:
            end = begin + i.size
            if i.need_grad == True:
                i.d = inputs0[begin:end].reshape(i.shape)
            begin = end

    # Gradient penalty on grads used for computing numerical gradients
    def obj_func(inputs0, gp2, vinputs):
        set_inputs(inputs0, vinputs)
        gp2.forward()
        return gp2.d.copy()

    # # Half test
    # if not disable_half_test:
    #     finputs = create_variables(inputs, backward)
    #     hinputs = create_variables(inputs, backward)
    #     half_test(rng, func, finputs, hinputs, func_args,
    #               func_kwargs, backward, ctx, func_name, atol=atol_half)

    # Create input variables
    vinputs = create_variables(inputs, backward)
    # --- Forward test --- #
    # Zero grads
    zero_grads(vinputs)
    # Forward/Backward on the forward graph
    voutputs = [F.sigmoid(x) for x in force_list(
        func(*(vinputs + func_args), **func_kwargs))]
    agrad_outputs = create_grad_outputs(voutputs)
    o = prod_sum(voutputs, agrad_outputs)
    o.forward()
    o.backward()  # clear_buffer=True)
    # Grads
    voutputs = voutputs
    vinputs = list(filter(lambda vi: vi is not None, vinputs))
    agrad_outputs = agrad_outputs
    grads = nn.grad(voutputs, vinputs, agrad_outputs)
    grads = list(filter(lambda x: x is not None, grads))
    o = F.sink(*grads)
    o.forward()
    # Check forward
    for vi, go in zip(vinputs, grads):
        if vi.need_grad is False:
            continue
        fgrads = vi.g
        bgrads = go.d
        assert_allclose(fgrads, bgrads, atol=atol_f)

    # TODO: 1. Pass function argument directly to backward functions.
    # TODO: 2. should be changed for the simplier form by simply testing BackwardFunction

    # --- Backward (accum = False) test --- #
    # Zero grads
    zero_grads(vinputs)
    # Compute analytical grads
    gp2 = gradient_penalty2(grads)
    gp2.forward()
    gp2.backward(clear_buffer=True)
    analytical_grads = np.concatenate(
        [vi.g.copy().flatten() for vi in vinputs])
    analytical_grads0 = analytical_grads
    # Compute numerical grads
    inputs0 = np.concatenate([inp.flatten()
                              for inp in inputs if inp is not None])
    numerical_grads = approx_fprime(inputs0, obj_func, dstep, gp2, vinputs)
    # Check backward
    assert_allclose(analytical_grads, numerical_grads, atol=atol_b)

    # --- Backward (accum = True) test --- #
    # Random grads
    rand_grads = [randn(rng, *vi.shape) for vi in vinputs]
    fill_grads(vinputs, rand_grads)
    # Compute analytical grads
    gp2.forward()
    gp2.backward(clear_buffer=True)

    analytical_grads = np.concatenate(
        [vi.g.copy().flatten() for vi in vinputs])
    rand_grads = np.concatenate([rg.flatten() if isinstance(rg, np.ndarray) else np.array(rg).reshape((1, ))
                                 for rg in rand_grads])
    analytical_grads -= rand_grads
    # Check backward
    assert_allclose(
        analytical_grads, analytical_grads0, atol=atol_accum)
