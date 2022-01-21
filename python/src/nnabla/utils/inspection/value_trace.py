# Copyright 2020,2021 Sony Corporation.
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

from __future__ import absolute_import

from collections import OrderedDict
from contextlib import contextmanager

import nnabla.functions as F

from .base import FunctionHookCallbackBase


def _number_to_order(n):
    if n == 3:
        return "3rd"
    elif n == 2:
        return "2nd"
    elif n == 1:
        return "1st"
    else:
        return "{}th".format(n)


def _error_trace(history, exec_name):
    print("Error during {} propagation".format(exec_name))
    for i in range(len(history) - 1):
        print("\t{}".format(history[i]))

    print("\t{} <-- ERROR".format(history[-1]))


class NanInfTracer(FunctionHookCallbackBase):
    """
    An utility API to create function_hook callbacks to check whether the outputs of all layers have NaN or inf as their values.
    During forward and backward execution, passed as function_hook,
    this API reports ValueError if at least one of all layer outputs has Nan or inf as its values.
    Otherwise, all tensors passed to next layer or function as is.

    Example:

    .. code-block:: python

        pred = model(...)

        from nnabla.utils.inspection import NanInfTracer
        nit = NanInfTracer(trace_inf=True, trace_nan=True, need_details=True)

        with nit.trace():
            pred.forward(function_post_hook=nit.forward_post_hook)
            pred.backward(function_post_hook=nit.backward_post_hook)
    """

    def __init__(self, trace_nan=True, trace_inf=True, need_details=True):
        super(NanInfTracer, self).__init__()

        self.trace_nan = trace_nan
        self.trace_inf = trace_inf
        self.need_details = need_details
        self.key_to_stat_fwd = OrderedDict()
        self.key_to_stat_bwd = OrderedDict()

        _msg_general = "The {} output of the function '{}' (rank: {}) has nan or inf as its values."
        _msg_detail = """
        Function details: 
            function type: {}
            shapes of inputs: {} 
            shapes of outputs: {}
            function args: {}
        """ if self.need_details else ""

        self._msg = _msg_general + _msg_detail
        self._msg_keys = ["name", "rank", "function_type",
                          "input_shapes", "output_shapes", "function_args"]

    def _add_key(self, f, key_to_stat):
        if f not in key_to_stat:
            key_to_stat[f] = {
                "name": f.name,
                "function_type": None, "function_args": None,
                "input_shapes": None, "output_shapes": None,
            }

            if self.need_details:
                key_to_stat[f].update({
                    "function_type": f.info.type_name,
                    "function_args": str(f.info.args),
                    "input_shapes": [str(x.shape) for x in f.inputs],
                    "output_shapes": [str(x.shape) for x in f.outputs],
                })

    @property
    def pre_hook(self):
        # Perform nothing for pre_hook.
        return None

    @property
    def post_hook(self):
        raise NotImplementedError("NanInfTracer has no member named post_hook. "
                                  "Please use forward_post_hook and backward_post_hook for forward and backward respectively instead.")

    @property
    def forward_post_hook(self):
        """
        Create callback function object which can be used as a function_post_hook argument of forward().
        """
        if not (self.trace_nan or self.trace_inf):
            return None

        # Perform F.isnan and then F.sum to check the output of incoming function contains the nan value.
        def callback(f):
            # For the first time to check this function.
            self._add_key(f, self.key_to_stat_fwd)

            # apply callback to check the outputs of this function has nan values or not.
            nan = []
            if self.trace_nan:
                nan = [F.sum(F.isnan(o.data)) for o in f.outputs]

            inf = []
            if self.trace_inf:
                inf = [F.sum(F.isinf(o.data)) for o in f.outputs]

            self.key_to_stat_fwd[f].update({
                "inf": inf,
                "nan": nan,
                # rank might be changed between each iteration.
                "rank": f.rank,
            })

        return callback

    @property
    def backward_post_hook(self):
        """
        Create callback function object which can be used as a function_post_hook argument of backward().
        """
        if not (self.trace_nan or self.trace_inf):
            return None

        # Perform F.isnan and then F.sum to check the output of incoming function contains the nan value.
        def callback(f):
            # For the first time to check this function.
            self._add_key(f, self.key_to_stat_bwd)

            # apply callback to check the outputs of this function has nan values or not.
            nan = []
            if self.trace_nan:
                nan = [F.sum(F.isnan(i.grad)) for i in f.inputs]

            inf = []
            if self.trace_inf:
                inf = [F.sum(F.isinf(i.grad)) for i in f.inputs]

            self.key_to_stat_bwd[f].update({
                "inf": inf,
                "nan": nan,
                # rank might be changed between each iteration.
                "rank": f.rank,
            })

        return callback

    @contextmanager
    def trace(self):
        """
        Create context manager to check nan/inf existence by using with statement.
        Using this context manager, checking nan/inf is performed automatically just before exiting with scope.
        Unless you use this context manager, be sure to call .check() explicitly to check nan/inf.

        Example:

        .. code-block:: python

            nit = NanInfTracer()
            with nit.trace():
                pred.forward(function_post_hook=nit.forward_post_hook)
                pred.backward(function_post_hook=nit.backward_post_hook)
        """
        # No need to release any resources at the end of this context manager
        yield self
        self.check()

    def _check_impl(self, key_to_stat, exec_name):
        history = []
        index = None
        for f, stat in key_to_stat.items():
            # check nan value
            if self.trace_nan:
                for i, v in enumerate(stat["nan"]):
                    if v.data > 0:
                        index = i
                        break

            if index is None and self.trace_inf:
                for i, v in enumerate(stat["inf"]):
                    if v.data > 0:
                        index = i
                        break

            history.append(f)

            if index is not None:
                _error_trace(history, exec_name)
                raise ValueError(self._msg.format(_number_to_order(index),
                                                  *[stat[x] for x in self._msg_keys]))

    def check(self):
        """
        Checks nan/inf existence at all outputs of all layers and raises ValueError only if exist.
        """
        # check forward
        self._check_impl(self.key_to_stat_fwd, "forward")

        # check backward
        self._check_impl(self.key_to_stat_bwd, "backward")
