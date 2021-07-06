# Copyright 2021 Sony Corporation.
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


from .tensor_normalization import tensor_normalization_backward


def instance_normalization_backward(inputs, channel_axis=None, batch_axis=(0,), eps=1e-05, no_scale=False, no_bias=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    x = inputs[1]

    # Reduce [H, W]
    axes = list(set(range(x.ndim)) - set(batch_axis + [channel_axis]))
    dx, _ = tensor_normalization_backward(inputs, axes, eps, no_scale, no_bias)
    return dx
