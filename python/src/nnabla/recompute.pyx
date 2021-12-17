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


from contextlib import contextmanager
from recompute cimport *

def set_global_recompute(recompute):
    c_set_global_recompute(recompute)

def get_global_recompute():
    return c_get_global_recompute()


@contextmanager
def recompute(recompute_=True):
    """Recompute all variables inside of code block.

    Variables inside `with nn.recompute():` block will be recomputed during backward propagation if possible.
    You can also specify boolean value switching recompute flag by `with nn.recompute(True):` or `with nn.recompute(False):`.
    `with` statement can be nested, so disabling recomputation part of the network is done by enclosing the partial network with `with nn.recompute(False):`.

    There is also function scope version of syntax for switching recompute flags. See `nn.recompute_fn()` documentation for more details.

    Args:

        recompute_ (bool): Re-computation flag. Default is True.

    Example:

    .. code-block:: python

        with nn.recompute():
            output0 = <Network0>(<input0>)

        output1 = <Network1>(<input1>, output0)
        loss = <Loss>(output1, <ground_truth>)
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)

        # All re-computable variables defined inside `with nn.recompute()` are recomputed during backward propagation.
        # In this case, variables defined inside `<Network0>` and `output0` are recomputed when possible.
        # Notice that the variable's `data` to be re-computed might have uninitialized values even after forward propagation since the `data` will be cleared during forward execution. Thus, you should not access the `data` of variables to be re-computed.
    """

    prev_recompute = get_global_recompute()
    set_global_recompute(recompute_)
    try:
        yield
    finally:
        set_global_recompute(prev_recompute)


def recompute_fn(recompute_=True):
    """Recompute all variables inside of function.

    Variables inside decorated functions with `@nn.recompute_fn()` will be recomputed during backward propagation if possible.
    You can also specify boolean value switching recompute flag by `@nn.recompute_fn(True)` or `@nn.recompute_fn(False)`.

    There is also block scope version of syntax for switching recompute flags. See `nn.recompute()` documentation for more ditails.

    Args:

        recompute_ (bool): Re-computation flag. Default is True.

    Example:

    .. code-block:: python

        @nn.recompute_fn()
        def network(x):
            y = <Functions>(x)
            return y

        output1 = network(<input1>, output0)
        loss = <Loss>(output1, <ground_truth>)
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)

        # All re-computable variables defined inside `network()` are recomputed during backward propagation.
        # Notice that the variable's `data` to be re-computed might have uninitialized values even after forward propagation since the `data` will be cleared during forward execution. Thus, you should not access the `data` of variables to be re-computed.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with recompute(recompute_):
                return func(*args, **kwargs)
        return wrapper
    return decorator
