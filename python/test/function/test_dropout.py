# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

from six.moves import range
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose


ctxs = list_context('Dropout')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("p", [-0.01, 1])
def test_dropout_p_boundaries(p, ctx, func_name):
    with nn.context_scope(ctx):
        x = nn.Variable((2, 3))
        with pytest.raises(RuntimeError):
            # Dropout cannot take p < 0 and 1 <= 1.
            y = F.dropout(x, p)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [p / 10. for p in range(1, 9)] + [0])
def test_dropout_forward_backward(p, seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region
    # Note: each backward execution requires a forward execution in NNabla.

    with nn.context_scope(ctx):
        # Create inputs
        rng = np.random.RandomState(seed)
        inputs = [
            cap_ignore_region(
                rng.randn(2, 3, 4).astype(np.float32) * 2,
                (-1e-3, 1e-3))]  # Ensure there is no zero.
        x = nn.Variable(inputs[0].shape, need_grad=True)
        x.d = inputs[0]
        init_dx = rng.randn(*x.shape).astype(x.data.dtype)
        init_dy = rng.randn(*x.shape).astype(x.data.dtype)

        # Construct graph
        y = F.dropout(x, p)

        # Reference parameter
        scale = 1. / (1. - p)

        # Test forward
        y.forward(clear_buffer=True)
        mask = (y.d != 0)
        ref_y = x.d * mask * scale
        assert_allclose(y.d, ref_y)
        assert y.parent.name == func_name

        # Test backward
        x.g[...] = init_dx
        y.backward(init_dy, clear_buffer=True)
        ref_dx = init_dy * mask * scale
        assert_allclose(x.g, init_dx + ref_dx)

        # Test accumulation
        y.forward(clear_no_need_grad=True)
        mask = (y.d != 0)
        x.g[...] = 1
        y.g = init_dy
        y.parent.backward([x], [y], [False])
        ref_dx = init_dy * mask * scale
        assert_allclose(x.g, ref_dx)

        # Test accum=False with NaN gradient
        y.forward(clear_no_need_grad=True)
        x.g = np.float32('nan')
        y.parent.backward([x], [y], [False])
        assert not np.any(np.isnan(x.g))

        # Test need_grad
        y.forward(clear_no_need_grad=True)
        x.g[...] = 0
        x.need_grad = False
        y.backward(init_dy)
        assert np.all(x.g == 0)


def ref_dropout_backward(dy, mask, p):
    return dy * mask / (1 - p)


def ref_dropout_double_backward(dy, mask, p):
    ''' This function returns the reference result of the derivative of 
        y + dy/dx where y = F.Dropout. Because the second derivative of Dropout
        is 0, this returns y' + 0.
    '''
    return ref_dropout_backward(dy, mask, p)  # + np.zeros(dy.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [p / 10. for p in range(1, 9)] + [0])
def test_dropout_double_backward(p, seed, ctx, func_name):
    from nnabla.backward_functions import registry
    from nnabla._dropout_workaround import _get_dropout_mask
    # dropout_backward depends on Dropout. The dependency must be kept by the
    # the execution order.
    # 1. Dropout::forward                  (A mask of dropout is calculated.)
    # 2. The forward of dropout_backward   (The mask is used.)
    # 3. The backward of dropout_backward  (The mask is used.)
    # 4. Dropout::backward                 (The mask is used, and then cleared.)
    # This order must be kept when using nnabla.grad. In the current
    # implementation, GradEndFunction keeps this order.
    atol_f = 1e-4

    with nn.context_scope(ctx):
        rng = np.random.RandomState(seed)
        init_x = rng.randn(2, 3, 4).astype(np.float32) * 2
        init_dy = rng.randn(*init_x.shape).astype(init_x.dtype)
        init_dy_for_grad = rng.randn(*init_x.shape).astype(init_x.dtype)
        init_dx = rng.randn(*init_x.shape).astype(init_x.dtype)
        init_for_dx2 = rng.randn(*init_x.shape).astype(init_x.dtype)

        #
        # A. Test mask passing
        #
        # Skip p=0 because, in the case, dropout does not happen. mask does not
        # change the results.
        if p != 0:
            with pytest.raises(RuntimeError):
                x = nn.Variable.from_numpy_array(init_x).apply(need_grad=True)
                dy = nn.Variable.from_numpy_array(
                    init_dy).apply(need_grad=True)
                # y = F.dropout(x, p, seed)  # Dropout is required to compute mask.
                dx = registry['Dropout']([dy, x], p, seed)

            # Note: y.forward() is required for dx.forward(). However this test
            #       is skipped because the random results are randomly matched
            #       between dx.forward() with and without y.forward(). Therefore
            #       The test result is not reproduced.

        #
        # B. Unit test of dropout_backward
        #
        # Graph construction
        x = nn.Variable.from_numpy_array(init_x).apply(need_grad=True)
        dy = nn.Variable.from_numpy_array(init_dy).apply(need_grad=True)
        y = F.dropout(x, p, seed)  # Dropout is required to compute mask.
        dx = registry['Dropout']([dy, x], p, seed)

        # Execution
        y.forward()  # Dropout is required to compute mask.
        # (y!=0) cannot be used when x includes 0.
        mask = _get_dropout_mask(x).d
        dx.forward()
        # Note: dropout_backward is a composite function. dx.parent is just
        #       a just composing function like MulScalar. Unit tests using
        #       dx.parent.forward and dx.parent.backward are meaningless.
        #       By the same reason, test of accumulation is nonsense.

        # Reference
        ref_dx = ref_dropout_backward(init_dy, mask, p)

        # Test
        assert_allclose(dx.d, ref_dx, atol=atol_f,
                        err_msg="Wrong output values of dropout_backward.")

        #
        # C. Test the forward of dropout_backward by using nnabla.grad
        #
        # Graph construction
        x = nn.Variable.from_numpy_array(init_x).apply(need_grad=True)
        y = F.dropout(x, p, seed)
        dx = nn.grad(y, x, grad_outputs=[init_dy_for_grad])[0]
        # Note: In NNabla 1.22.0, if use grad_outputs=X, nn.grad separate
        #       np.ndarray X into small arrays by self._force_list.
        #       For example, X = np.array([[5, 6], [7, 8]]) is separated
        #       into [np.array([5, 6]), np.array(7, 8)]. Then Mul2 inserted by
        #       nn.grad uses np.array([5, 6]) as dy, and broadcasts it to
        #       the np.array([[5, 6], [5, 6]]). Finally, the forward execution
        #       is finished, but the result values are wrong.

        # Execution
        dx.forward(clear_buffer=True)

        # Reference
        mask = _get_dropout_mask(x).d
        ref_dx = ref_dropout_backward(init_dy_for_grad, mask, p)

        # Test
        assert_allclose(dx.d, ref_dx, atol=atol_f,
                        err_msg="Wrong output values of Dropout of nn.grad.")

        #
        # D. Test the backward of dropout_backward by using nnabla.grad
        #
        # The numerical grad by using scipy.approx_fprime cannot be performed
        # because Dropout has randomness and changes the results during
        # the repeated forward computation.

        # Graph construction
        x = nn.Variable.from_numpy_array(init_x).apply(need_grad=True)
        y = F.dropout(x, p, seed)
        dx = nn.grad(y, x, grad_outputs=[init_dy_for_grad])[0]
        y_dx = y + dx  # replaceable with F.sink(y, dx, one_input_grad=False)

        # Execution
        x.g = init_dx  # Accumulation
        y_dx.forward(clear_no_need_grad=True)
        mask = _get_dropout_mask(x).d  # Store mask before the clear
        y_dx.backward(init_for_dx2, clear_buffer=True)

        # Reference
        ref_dx = ref_dropout_double_backward(init_for_dx2, mask, p) + init_dx

        # Test
        assert_allclose(x.g, ref_dx, atol=atol_f,
                        err_msg="Wrong output values of double backward of "
                                "Dropout by nn.grad.")

        #
        # E. Test the backward with and without accumulation
        #
        # Let dx = dropout_backward(dy). Because dropout_backward is implemented
        # as a composite function, dx.parent cannot determine the backward
        # function of dropout_backward. Therefore the tests of accumulation
        # by using dx.parent(..., accum=[False]) cannot be performed here.
        # It is not problem because the accumulation of each composing function
        # is expected to be tested independently.
        #
        # Note: Under the depth-first search of NNabla graph engine,
        #       GradEndFunction determines the order of accumulation such that
        #         x.g += (the backward propagation from y.g)
        #         x.g += (the backward propagation from dx.g).
        #       So the test D could fail when accumulation of the
        #       double-backward path fails.


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [0.5])
def test_dropout_grad_dependency(p, seed, ctx, func_name):
    from nnabla._dropout_workaround import _get_dropout_mask
    # Test whether the memory clearance by grad_depends_on_inputs/outputs does
    # something bad during graph execution such as the clearance values which
    # is planned to be used. This test is performed by changing the
    # inputs/outputs of Dropout to intermediate variables in the same manner of
    # nbla_test_utils.py.
    atol_f = 1e-4

    with nn.context_scope(ctx):
        rng = np.random.RandomState(seed)
        init_x = rng.randn(2, 3, 4).astype(np.float32) * 2
        init_dy_for_grad = rng.randn(*init_x.shape).astype(init_x.dtype)
        init_dx = rng.randn(*init_x.shape).astype(init_x.dtype)
        init_for_dx2 = rng.randn(*init_x.shape).astype(init_x.dtype)

        # Graph construction
        x = nn.Variable.from_numpy_array(init_x).apply(need_grad=True)
        x_interm = F.identity(x)
        y_interm = F.dropout(x_interm, p, seed)
        y = F.identity(y_interm)
        dx_interm = nn.grad(y, x, grad_outputs=[init_dy_for_grad])[0]
        dx = F.identity(dx_interm)
        y_dx = y + dx  # replaceable with F.sink(y, dx, one_input_grad=False)

        # Execution
        x.g = init_dx  # Accumulation
        y_dx.forward(clear_no_need_grad=True)
        mask = _get_dropout_mask(x_interm).d  # Store mask before the clear
        y_dx.backward(init_for_dx2, clear_buffer=True)

        # Reference
        ref_dx = ref_dropout_double_backward(init_for_dx2, mask, p) + init_dx

        # Test
        assert_allclose(x.g, ref_dx, atol=atol_f,
                        err_msg="Wrong output values of double backward of "
                                "Dropout by nn.grad.")


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [-1, 313])
@pytest.mark.parametrize("p", [0.5] + [0])
def test_dropout_recompute(p, seed, ctx, func_name):
    from nbla_test_utils import recomputation_test

    rng = np.random.RandomState(0)
    x = nn.Variable((2, 3, 4))
    func_args = [p, seed]
    recomputation_test(rng=rng, func=F.dropout, vinputs=[x],
                       func_args=func_args, func_kwargs={}, ctx=ctx)
