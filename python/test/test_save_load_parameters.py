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

from six import iteritems

import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF


def test_save_load_parameters():
    v = nn.Variable([64, 1, 28, 28], need_grad=False)
    with nn.parameter_scope("param1"):
        with nn.parameter_scope("conv1"):
            h = PF.convolution(v, 32, (3, 3))
            b = PF.batch_normalization(h, batch_stat=True)
        with nn.parameter_scope("conv2"):
            h1 = PF.convolution(v, 32, (3, 3))
            b2 = PF.batch_normalization(h1, batch_stat=True)

    for k, v in iteritems(nn.get_parameters(grad_only=False)):
        v.data.cast(np.float32)[...] = np.random.randn(*v.shape)

    with nn.parameter_scope("param1"):
        param1 = nn.get_parameters(grad_only=False)
        nn.save_parameters("tmp.h5")
        nn.save_parameters("tmp.protobuf")

    with nn.parameter_scope("param2"):
        nn.load_parameters('tmp.h5')
        param2 = nn.get_parameters(grad_only=False)

    with nn.parameter_scope("param3"):
        nn.load_parameters('tmp.protobuf')
        param3 = nn.get_parameters(grad_only=False)

    for par2 in [param2, param3]:
        assert param1.keys() == par2.keys()  # Check order
        for (n1, p1), (n2, p2) in zip(sorted(param1.items()), sorted(par2.items())):
            assert n1 == n2
            assert np.all(p1.d == p2.d)
            if par2 is not param3:
                # NOTE: data is automatically casted to fp32 in Protobuf
                assert p1.data.dtype == p2.data.dtype
            assert p1.need_grad == p2.need_grad
