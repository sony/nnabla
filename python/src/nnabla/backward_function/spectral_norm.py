# Copyright 2021 Sony Corporation.
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


import numpy as np
import nnabla as nn
import nnabla.functions as F
from .utils import no_grad, sum_for_arithmetics


def _spectral_norm_backward(dw_sn, w, u, dim=0, itr=1, eps=1e-12):
    # Forward recomputation

    w_shape = w.shape
    # Transpose if the output dimension is not the most-left dimension.
    if dim != 0:
        dims_transpose = [dim] + [i for i in range(len(w_shape)) if i != dim]
        w = F.transpose(w, dims_transpose)
        w_shape = w.shape
    d0 = w.shape[0]            # Out
    d1 = np.prod(w.shape[1:])  # In
    w = F.reshape(w, [d0, d1])
    u = F.reshape(u, [1, d0])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(u, w)
        v = v / ((F.sum(v ** 2.0, keepdims=True) + eps) ** 0.5)
        v = F.reshape(v, [d1, 1])
        # u
        u = F.affine(w, v)
        u = u / ((F.sum(u ** 2.0, keepdims=True) + eps) ** 0.5)
        u = F.reshape(u, [1, d0])
    # No grad
    u = no_grad(u)
    v = no_grad(v)
    # Spectral normalization
    wv = F.affine(w, v)
    sigma = F.affine(u, wv)
    w_sn = w / sigma
    # The fowllowing process is not necessary for gradient calculation
    # w_sn = F.reshape(w_sn, w_shape)
    # # Transpose again if the output dimension is not the most-left dimension.
    # if dim != 0:
    #     dims_transpose = [i for i in range(1, dim + 1)] \
    #                      + [0] + [i for i in range(dim + 1, len(w_shape))]
    #     w_sn = F.transpose(w_sn, dims_transpose)

    # Backward

    # Backward for post-transpose
    if dim != 0:
        dims_transpose = [dim] + [i for i in range(len(w_shape)) if i != dim]
        dw_sn = F.transpose(dw_sn, dims_transpose)
    dw_sn = dw_sn.reshape(w.shape)

    # Backward for spectral norm
    # Sum for broadcast backward
    S = sum_for_arithmetics(dw_sn * w_sn, sigma)
    # Add batch axis
    S = S.reshape((1,) + S.shape)
    u = u.reshape((1,) + u.shape)
    v = v.reshape((1,) + v.shape)
    m = F.batch_matmul(u, S, transpose_a=True)
    m = F.batch_matmul(m, v, transpose_b=True)
    # Remove batch axis
    m = m.reshape((m.shape[1], m.shape[2]))
    dw = (dw_sn - m) / sigma

    # Backward for pre-transpose
    dw = dw.reshape(w_shape)
    if dim != 0:
        dims_transpose = [i for i in range(1, dim + 1)] \
                         + [0] + [i for i in range(dim + 1, len(w_shape))]
        dw = F.transpose(dw, dims_transpose)

    return dw, None


def _spectral_norm_outer_most_dim_backward(dw_sn, w, u, itr=1, eps=1e-12):
    # Forward recomputation

    w_shape = w.shape
    d0 = np.prod(w.shape[0:-1])  # In
    d1 = w.shape[-1]             # Out
    w = F.reshape(w, [d0, d1])
    u = F.reshape(u, [d1, 1])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(w, u)
        v = v / ((F.sum(v ** 2.0, keepdims=True) + eps) ** 0.5)
        v = F.reshape(v, [1, d0])
        # u
        u = F.affine(v, w)
        u = u / ((F.sum(u ** 2.0, keepdims=True) + eps) ** 0.5)
        u = F.reshape(u, [d1, 1])
    # No grad
    u = no_grad(u)
    v = no_grad(v)
    # Spectral normalization
    vw = F.affine(v, w)
    sigma = F.affine(vw, u)
    w_sn = w / sigma
    # The fowllowing process is not necessary for gradient calculation
    # w_sn = F.reshape(w_sn, w_shape)

    # Backward for spectral norm
    dw_sn = dw_sn.reshape(w.shape)
    # Sum for broadcast backward
    S = sum_for_arithmetics(dw_sn * w_sn, sigma)
    # Add batch axis
    S = S.reshape((1,) + S.shape)
    u = u.reshape((1,) + u.shape)
    v = v.reshape((1,) + v.shape)
    m = F.batch_matmul(v, S, transpose_a=True)
    m = F.batch_matmul(m, u, transpose_b=True)
    # Remove batch axis
    m = m.reshape((m.shape[1], m.shape[2]))
    dw = (dw_sn - m) / sigma
    dw = dw.reshape(w_shape)

    return dw, None


def spectral_norm_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, dim=0, itr=1, eps=1e-12, test=False, output_u=False):
    """
    Args:
      grad_inputs (list of :obj:`nnabla.Variable`): Propagated grads to this backward function.
      inputs (list of :obj:`nnabla.Variable` and None): Input Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      input_shapes (list of tuple of :obj:`int`): Input shapes of the forward function.
          The shapes of the inputs in which None is set can be passed.
      outputs (list of :obj:`nnabla.Variable` and None): Output Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      output_shapes (list of tuple of :obj:`int`): Output shapes of the forward function.
          The shapes of the outputs in which None is set can be passed.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    if not output_u:
        # We need to get original `u` from output for gradient calculation.
        raise ValueError(
            "spectral_norm_backward is supported for output_u=True.")

    dw_sn = grad_inputs[0]
    w = inputs[0]
    u = outputs[1]

    if dim == w.ndim - 1:
        return _spectral_norm_outer_most_dim_backward(dw_sn, w, u, itr, eps)
    else:
        return _spectral_norm_backward(dw_sn, w, u, dim, itr, eps)
