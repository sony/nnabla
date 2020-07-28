# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nnabla as nn
import nnabla.functions as F
from nnabla.utils.rnn import _create_fixed_length_lstm


def lstm_backward(inputs, num_layers=1, dropout=None, bidirectional=False, training=True):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    if dropout != 0.0:
        raise ValueError("Dropout must be 0.0")

    dys = inputs[0]
    dhn = inputs[1]
    dcn = inputs[2]
    xs0 = inputs[3]
    h0 = inputs[4]
    c0 = inputs[5]
    w0 = inputs[6]

    if num_layers == 1:
        w = None
        b = inputs[7] if len(inputs) == 8 else None
    else:
        w = inputs[7]
        b = inputs[8] if len(inputs) == 9 else None
    num_directions = 2 if bidirectional else 1
    with_bias = True if b else False

    ys, hn, cn = _create_fixed_length_lstm(
        xs0, h0, c0, w0, w, b, num_layers, num_directions, with_bias)
    outputs = [ys, hn, cn]
    grad_outputs = [dys, dhn, dcn]
    if w and b:
        inputs = [xs0, h0, c0, w0, w, b]
        dxs0, dh0, dc0, dw0, dw, db = nn.grad(
            outputs, inputs, grad_outputs=grad_outputs)
        return dxs0, dh0, dc0, dw0, dw, db
    if w and not b:
        inputs = [xs0, h0, c0, w0, w]
        dxs0, dh0, dc0, dw0, dw = nn.grad(
            outputs, inputs, grad_outputs=grad_outputs)
        return dxs0, dh0, dc0, dw0, dw
    if not w and b:
        inputs = [xs0, h0, c0, w0, b]
        dxs0, dh0, dc0, dw0, db = nn.grad(
            outputs, inputs, grad_outputs=grad_outputs)
        return dxs0, dh0, dc0, dw0, db
    if not w and not b:
        inputs = [xs0, h0, c0, w0]
        dxs0, dh0, dc0, dw0 = nn.grad(
            outputs, inputs, grad_outputs=grad_outputs)
        return dxs0, dh0, dc0, dw0
