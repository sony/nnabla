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

from nnabla.utils.inspection import TimeProfiler, pprint

from .models import simple_cnn


@pytest.mark.parametrize("batch_size, n_class", [(8, 10)])
@pytest.mark.parametrize("forward, backward", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("summary", [True, False])
@pytest.mark.parametrize("hidden", [True, False])
@pytest.mark.parametrize("printer", [True, False])
@pytest.mark.parametrize("ext_name", [x for x in list_extensions() if x in ["cpu", "cudnn"]])
def test_pprint(batch_size, n_class, forward, backward, summary, hidden, printer, ext_name):
    nn.clear_parameters()
    ctx = get_extension_context(ext_name)

    x = nn.Variable.from_numpy_array(
        np.random.normal(size=(batch_size, 3, 16, 16)))
    t = nn.Variable.from_numpy_array(np.random.randint(low=0, high=n_class,
                                                       size=(batch_size, 1)))
    with nn.context_scope(ctx):
        y = simple_cnn(x, t, n_class)

    pprint(y, forward, backward, summary, hidden, printer)
