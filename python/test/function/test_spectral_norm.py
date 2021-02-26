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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nbla_test_utils import list_context

ctxs = list_context('SpectralNorm')


def ref_spectral_norm(w, u, dim=0, itr=1, eps=1e-12, test=False):
    w_shape = w.shape
    if dim != 0:
        dims_transpose = [dim] + \
            [i for i in range(len(w_shape)) if i != dim]
        w = w.transpose(*dims_transpose)
        w_shape = w.shape
    d0, d1 = w_shape[0], np.prod(w_shape[1:])  # [Out, In]
    w = w.reshape((d0, d1))
    for i in range(itr):
        v = np.dot(w.T, u)
        v = v / np.sqrt(np.sum(v ** 2) + eps)
        u = np.dot(w, v)
        u = u / np.sqrt(np.sum(u ** 2) + eps)
    sigma = np.dot(u.T, np.dot(w, v))
    w_sn = w / sigma
    w_sn = w_sn.reshape(w_shape)
    if dim != 0:
        dims_transpose = [i for i in range(1, dim + 1)] \
                            + [0] + [i for i in range(dim + 1, len(w_shape))]
        w_sn = w_sn.transpose(*dims_transpose)
    return w_sn


def ref_grad_spectral_norm(w, u, dy, dim, itr, eps, test, need_grad_flags):
    # We need this function for using `function_tester`
    # because the numerical gradient of `w` will not be calculated correctly.
    # The reason is there are some intermediate variables with `need_grad == false`
    # which are connected to the input `w` in the function composite implementation.

    cpu_context = nn.Context(["cpu:float"])
    with nn.context_scope(cpu_context):
        w = nn.Variable.from_numpy_array(w)
        u = nn.Variable.from_numpy_array(u)
        w.need_grad = True
        w.grad.zero()
        w_sn = PF._spectral_norm_v1(
            w, u_init=u.data.get_data('r'), dim=dim, itr=itr, test=test)

        w_sn.forward(clear_no_need_grad=True)
        w_sn.backward(dy, clear_buffer=True)

    return w.grad.get_data('r').flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("w_shape, dim", [((32, 16, 3, 3), 0),  # convolution
                                          ((16, 1), 1),         # affine
                                          ((16, 32), 1),        # affine
                                          ((8, 8, 16), 2),      # affine
                                          ((8, 4, 16), 1),      # affine
                                          ])
@pytest.mark.parametrize("itr", [1, 2, 3])
@pytest.mark.parametrize("eps", [1e-12])
@pytest.mark.parametrize("test", [False, True])
@pytest.mark.parametrize("seed", [313])
def test_spectral_norm_forward_backward(seed, test, eps, itr, w_shape, dim, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    w = rng.randn(*w_shape).astype(np.float32)
    u = rng.randn(*(w_shape[dim],)).astype(np.float32)
    inputs = [w, u]

    backward = [False, False] if test else [True, False]

    function_tester(rng, F.spectral_norm, ref_spectral_norm, inputs, func_args=[dim, itr, eps, test],
                    backward=backward, ref_grad=ref_grad_spectral_norm, atol_accum=2e-2, ctx=ctx, func_name=func_name)
