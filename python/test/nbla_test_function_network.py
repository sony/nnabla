import nnabla as nn
import numpy as np
from scipy import optimize
from nnabla.testing import assert_allclose


def _randn(rng, *shape):
    return np.asarray(rng.randn(*shape), dtype=np.float32)


def _create_variables(inputs, backward):
    variable_inputs = []

    for i, b in zip(inputs, backward):
        if not isinstance(i, np.ndarray):
            variable_inputs += [nn.Variable(need_grad=b)]
            variable_inputs[-1].data.cast(type(i))[...] = i
        else:
            variable_inputs += [nn.Variable(i.shape, need_grad=b)]
            variable_inputs[-1].data.cast(i.dtype)[...] = i

    return variable_inputs


def compute_nnabla_and_numerical_grad(inputs, variable_inputs, variable_outputs, rng=None, epsilon=1e-8, have_scalar=False):
    '''
    Compute both nnabla grad and numerical grad using given functions

    Args:
        inputs: ndarray inputs
        variable_inputs: function input variables
        variable_outputs: function output variables
        rng: random number generator
        epsilon: small value to calculate numerical grad
    Returns:

    '''

    if rng is None:
        rng = np.random.RandomState(np.random.randint(1000))

    for i in variable_inputs:
        i.g = _randn(rng, *i.shape)

    def func(vector):
        bind = 0
        backups = []

        for vi, i in zip(variable_inputs, inputs):
            if not have_scalar:
                vi.d[...] = np.reshape(vector[bind: bind + vi.size], vi.shape)
                bind += vi.size
            else:
                if not isinstance(i, np.ndarray):
                    vi.d[...] = np.reshape(vector[bind: bind + 1], vi.shape)
                    bind += 1
                else:
                    vi.d[...] = np.reshape(
                        vector[bind: bind + vi.size], vi.shape)
                    bind += vi.size

            backups.append(vi.d.copy())

        variable_outputs.forward()
        for index, vi in enumerate(variable_inputs):
            vi.d[...] = backups[index]

        return np.sum(1.0 * variable_outputs.d)

    def _nbla_grad(vector):
        bind = 0
        backups = []

        for vi, i in zip(variable_inputs, inputs):
            if not have_scalar:
                vi.d[...] = np.reshape(vector[bind: bind + vi.size], vi.shape)
                bind += vi.size
            else:
                if not isinstance(i, np.ndarray):
                    vi.d[...] = np.reshape(vector[bind: bind+1], vi.shape)
                    bind += 1
                else:
                    vi.d[...] = np.reshape(
                        vector[bind: bind + vi.size], vi.shape)
                    bind += vi.size

            backups.append(vi.g.copy())

        variable_outputs.forward()
        variable_outputs.backward()

        bind = 0
        g = np.zeros_like(vector)
        index = 0
        for vi, i in zip(variable_inputs, inputs):
            g[bind: bind + vi.size] = vi.g.flatten() - backups[index].flatten()
            vi.g[...] = backups[index]
            bind += vi.size
            index += 1

        return g

    vector = []
    if have_scalar:
        for i in inputs:
            if not isinstance(i, np.ndarray):
                vector += [i]
            else:
                vector += i.flatten().tolist()
    else:
        for i in inputs:
            vector += i.flatten().tolist()

    nbla_grad = _nbla_grad(vector)
    numerical_grad = optimize.approx_fprime(vector, func, epsilon)

    return nbla_grad, numerical_grad


def function_network_tester(rng, func, inputs, func_args=[], args_out=False, have_scalar=False, backward=None, atol_b=1e-3, dstep=1e-3):
    '''
    Automatic testing of backward of `func`

    Args:
        rng: random number generator
        func: test function name
        inputs: the inputs of func
        func_args: the other func args of func
        args_out: func_args whether contain output arguments, output arguments always set last
        have_scalar: whether the operands contain scalars
        backward: the attribute of nn.Variable
    '''

    inputs_ = inputs
    if have_scalar:
        if args_out:
            inputs_ += func_args[:-1]
            func_args = [func_args[-1]]
        else:
            inputs_ += func_args
            func_args = []

    if backward is None:
        backward = [True for _ in inputs_]

    variable_inputs = _create_variables(inputs_, backward)

    # checking backward
    if False in backward:
        return

    # NNabla backward
    for v in variable_inputs:
        v.g = _randn(rng, *v.shape)

    y = func(*(variable_inputs + func_args))
    if args_out:
        y = func_args[-1]

    nbla_grad, numerical_grad = compute_nnabla_and_numerical_grad(
        inputs_, variable_inputs, y, rng, epsilon=dstep, have_scalar=have_scalar)
    assert_allclose(nbla_grad, numerical_grad, atol=atol_b,
                    err_msg="{} backward w/o accumulation test fails.".format(func))
