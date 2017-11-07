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

from six.moves import filter
import pytest
import copy
import numpy as np
import nnabla as nn
import nnabla.function as F_
import nnabla.functions as F


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
    except:
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
                                          vgrads, epsilon=1e-8, rng=None):
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


def function_tester(rng, func, ref_func, inputs,
                    func_args=[], func_kwargs={},
                    atol_f=1e-6, atol_b=1e-3, atol_accum=1e-6, dstep=1e-3, backward=None,
                    ctx=None, func_name=None, ref_grad=None):
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
    # print 'create_variable'

    def create_variables(inputs, backward):
        vinputs = []
        for i, b in zip(inputs, backward):
            if i is None:
                vinputs += [None]
                continue
            vinputs += [nn.Variable(i.shape, need_grad=b)]
            vinputs[-1].data.cast(i.dtype)[...] = i
        return vinputs
    vinputs = create_variables(inputs, backward)

    # Checking forward
    # print 'checking forward'
    with nn.context_scope(ctx), nn.auto_forward():
        o = func(*(vinputs + func_args), **func_kwargs)
    rinputs = copy.deepcopy(inputs)  # inputs for ref_func
    refs = ref_func(*(rinputs + func_args), **func_kwargs)

    def force_tuple(x):
        if isinstance(x, tuple):
            return x
        return (x,)
    refs = force_tuple(refs)
    o = force_tuple(o)
    assert len(o) == len(refs)
    for i, ref in enumerate(refs):
        res = o[i].d
        assert np.allclose(ref, res, atol=atol_f), str(
            ArrayDiffStats(ref, res))

    # Checking function name
    # print 'checking function name'
    if func_name is not None:
        assert o[0].parent.name == func_name

    # Checking backward
    # print 'checking backward'
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
        rng=rng)
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
        g = np.random.randn(*v.shape)
        v.g = g
        f.forward(finputs, o)
        f.backward(finputs, o)
        true_g = v.g - g

        # Check accum=False
        accum = [j != i for j in range(len(finputs))]
        v.g = np.random.randn(*v.shape)
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
