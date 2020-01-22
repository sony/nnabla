# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

from nnabla.ext_utils import get_extension_context, list_extensions
from nnabla.utils.inspection import NanInfTracer

from .models import simple_cnn


def _refresh_inputs_grad(f):
    for i in f.inputs:
        i.grad.zero()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("n_class", [5])
@pytest.mark.parametrize("ext_name", list_extensions())
@pytest.mark.parametrize("trace_nan", [False, True])
@pytest.mark.parametrize("trace_inf", [False, True])
def test_nan_inf_tracer(batch_size, n_class, ext_name, trace_nan, trace_inf):
    nn.clear_parameters()

    ctx = get_extension_context(ext_name)
    nn.set_default_context(ctx)

    x = nn.Variable.from_numpy_array(
        np.random.normal(size=(batch_size, 3, 16, 16)))
    t = nn.Variable.from_numpy_array(np.random.randint(low=0, high=n_class,
                                                       size=(batch_size, 1)))

    y = simple_cnn(x, t, n_class)

    must_be_inf = y / F.constant(0, shape=y.shape)
    must_be_nan = must_be_inf / must_be_inf

    # Refresh all arrays once so as to ensure all grad values are 0.
    must_be_nan.visit(_refresh_inputs_grad)

    nit = NanInfTracer(trace_nan=trace_nan, trace_inf=trace_inf)

    # can be run at any cases without exception.
    with nit.trace():
        y.forward(clear_no_need_grad=True,
                  function_post_hook=nit.forward_post_hook)
        y.backward(clear_buffer=True,
                   function_post_hook=nit.backward_post_hook)

    nit.check()  # this call can also work without exception.

    # check nan
    if trace_nan:
        with pytest.raises(ValueError):
            with nit.trace():
                must_be_nan.forward(clear_buffer=True,
                                    function_post_hook=nit.forward_post_hook)

        with pytest.raises(ValueError):
            with nit.trace():
                must_be_nan.backward(clear_buffer=True,
                                     function_post_hook=nit.backward_post_hook)

        must_be_nan.forward(clear_buffer=True,
                            function_post_hook=nit.forward_post_hook)
        with pytest.raises(ValueError):
            nit.check()

        must_be_nan.backward(clear_buffer=True,
                             function_post_hook=nit.backward_post_hook)

        with pytest.raises(ValueError):
            nit.check()

    # check inf
    if trace_inf:
        with pytest.raises(ValueError):
            with nit.trace():
                must_be_inf.forward(clear_buffer=True,
                                    function_post_hook=nit.forward_post_hook)

        must_be_inf.forward(clear_buffer=True,
                            function_post_hook=nit.forward_post_hook)
        with pytest.raises(ValueError):
            nit.check()
