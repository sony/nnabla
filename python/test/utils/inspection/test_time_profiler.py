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

import nnabla as nn
import numpy as np

from nnabla.ext_utils import get_extension_context, list_extensions

from nnabla.utils.inspection import TimeProfiler

from .models import simple_cnn


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("n_class", [5])
@pytest.mark.parametrize("ext_name", [x for x in list_extensions() if x in ["cpu", "cudnn"]])
def test_time_profiler(batch_size, n_class, ext_name, tmpdir):
    nn.clear_parameters()

    ctx = get_extension_context(ext_name)
    nn.set_default_context(ctx)

    x = nn.Variable.from_numpy_array(
        np.random.normal(size=(batch_size, 3, 16, 16)))
    t = nn.Variable.from_numpy_array(np.random.randint(low=0, high=n_class,
                                                       size=(batch_size, 1)))

    y = simple_cnn(x, t, n_class)

    tp = TimeProfiler(ext_name, device_id=ctx.device_id)
    for i in range(5):
        with tp.scope("forward"):
            y.forward(clear_no_need_grad=True,
                      function_pre_hook=tp.pre_hook,
                      function_post_hook=tp.post_hook)

        with tp.scope("backward"):
            y.backward(clear_buffer=True,
                       function_pre_hook=tp.pre_hook,
                       function_post_hook=tp.post_hook)

        tp.calc_elapsed_time(["forward", "backward", "summary"])

    tp()
    tp.to_csv(out_dir=str(tmpdir))
