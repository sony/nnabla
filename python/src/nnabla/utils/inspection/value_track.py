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

from __future__ import absolute_import

from nnabla import forward_all
import nnabla.functions as F

from collections import OrderedDict
from contextlib import contextmanager

from .base import FunctionHookCallbackBase


def _number_to_order(n):
    if n == 3:
        return "3rd"
    elif n == 2:
        return "2nd"
    elif n == 1:
        return "1st"
    else:
        return f"{n}th"


class NanInfTracker(FunctionHookCallbackBase):
    """
    An utility API to create function_hook callbacks to check whether the outputs of all layers have NaN or inf as their values.
    During forward and backward execution, passed as function_hook,
    this API reports ValueError if at least one of all layer outputs have Nan or inf.
    Otherwise, all tensors passed to next layer or function as is.

    Example:
        .. code-block:: python

    """
    def __init__(self, track_nan=True, track_inf=True, need_details=True):
        super(NanInfTracker, self).__init__()

        self.track_nan = track_nan
        self.track_inf = track_inf
        self.need_details = need_details
        self.key_to_stat = OrderedDict()

        _msg_general = "The {} output of the function '{}' (rank: {}) has nan or inf as its values."
        _msg_detail = """
        Function details: 
            function type: {}
            shapes of inputs: {} 
            shapes of outpus: {}
            function args: {}
        """ if self.need_details else ""

        self._msg = _msg_general + _msg_detail
        self._msg_keys = ["name", "rank", "function_type", "input_shapes", "output_shapes", "function_args"]

    @property
    def pre_hook(self):
        # Perform nothing for pre_hook.
        return None

    @property
    def post_hook(self):
        if not (self.track_nan and self.track_inf):
            return None

        # Perform F.isnan and then F.sum to check the output of incoming function contains the nan value.
        def callback(f):
            # For the first time to check this function.
            if f not in self.key_to_stat:
                self.key_to_stat[f] = {
                    "name": f.name,
                    "function_type": None, "function_args": None,
                    "input_shapes": None, "output_shapes": None,
                }

                if self.need_details:
                    self.key_to_stat[f].update({
                        "function_type": f.info.type_name,
                        "function_args": str(f.info.args),
                        "input_shapes": [str(x.shape) for x in f.inputs],
                        "output_shapes": [str(x.shape) for x in f.outputs],
                    })

            # apply callback to check the outputs of this function has nan values or not.
            nan = []
            if self.track_nan:
                nan = [F.sum(F.isnan(o.get_unlinked_variable(need_grad=False))) for o in f.outputs]

            inf = []
            if self.track_inf:
                inf = [F.sum(F.isinf(o.get_unlinked_variable(need_grad=False))) for o in f.outputs]
            forward_all(nan + inf, clear_no_need_grad=True)

            self.key_to_stat[f].update({
                "inf": inf,
                "nan": nan,
                "rank": f.rank,  # rank might be changed between each iteration.
            })

        return callback

    @contextmanager
    def track(self):
        """
        Create context manager to check nan/inf existence by using with statement.
        Using this context manager, checking nan/inf is performed automatically just before exiting with scope.
        Unless you use this context manager, be sure to call .check() explicitly to check nan/inf.

        Example:
            .. code-block:: python
                nit = NanInfTracker()
                with nit.track():
                    pred.forward(function_post_hook=nit.post_hook)
                    pred.backward(function_post_hook=nit.post_hook)

        """
        # No need to release any resources at the end of this context manager
        yield self
        self.check()

    def check(self):
        """
        Checks nan/inf existence at all outputs of all layers..
        """
        for f, stat in self.key_to_stat.items():
            # check nan value
            if self.track_nan:
                for o_i, (v1, v2) in enumerate(zip(stat["nan"], stat["inf"])):
                    if v1.d > 0 or v2.d > 0:
                        raise ValueError(self._msg.format(_number_to_order(o_i),
                                                          *[stat[x] for x in self._msg_keys]))
