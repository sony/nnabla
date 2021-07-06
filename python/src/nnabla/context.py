# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

"""
NNabla Context manager
"""
from __future__ import absolute_import

from contextlib import contextmanager
from .variable import Context


current_ctx = Context()
context_level = 0


@contextmanager
def context_scope(ctx):
    """
    Context as Python context.

    .. code-block:: python

        import nnabla as nn
        import nnabla.functions as F
        x = nn.Variable([2, 3 ,4])
        ctx = nnabla_ext.cuda.context('0')
        with context_scope(ctx):
            # Inside with scope, the specified context is used.
            with parameter_scope('w1'):
                l1 = F.relu(F.affine(x, 64))
            with parameter_scope('w2'):
                l2 = F.relu(F.affine(x, 64))

    """
    global current_ctx
    global context_level
    context_level += 1
    prev_context = current_ctx
    current_ctx = ctx
    try:
        yield
    finally:
        context_level -= 1
        current_ctx = prev_context


def set_default_context(ctx):
    """
    Set the default context.

    Note:
        It cannot be called inside any `context_scope`.

    Args:
        ctx (Context): A Context.

    """
    global context_level
    global current_ctx
    assert context_level == 0, "It cannot be called inside any context_scope."
    current_ctx = ctx


def get_current_context():
    """
    Get the current context.

    It can be set using :meth:`nnabla.context_scope` or :meth:`nnabla.set_default_context` .

    Returns:
        Context: a current context.

    """
    global current_ctx
    return current_ctx
