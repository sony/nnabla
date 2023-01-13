# Copyright 2022 Sony Group Corporation.
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

import os
import pytest
import numpy as np
import nnabla as nn
import nnabla.utils.cli.optimize_model as opt_cli
from .nntxt import (N0007, N0008)
from .common import generate_case_from_nntxt_str
from nnabla.utils import load


cases = {
    'without_bias': N0007,
    'with_bias': N0008
}


inputs = [np.random.random((1, 32))]


def forward(info):
    for e in info.executors.values():
        for v, d in zip(e.dataset_assign.keys(), inputs):
            v.variable_instance.d = d

    for v, generator in e.generator_assign.items():
        v.variable_instance.d = generator(v.variable_instance.d.shape)

    e.forward_target.forward(clear_buffer=True)

    for o in e.output_assign.keys():
        yield o.variable_instance.d


@pytest.mark.parametrize("case_name", ["without_bias", "with_bias"])
def test_nnp_optimizer_affine(case_name):
    class Args:
        pass
    args = Args()
    nnp_file_name = f"{case_name}.nnp"
    with generate_case_from_nntxt_str(cases[case_name], nnp_file_name, ".h5", 32) as nnp_file:
        args.input_file = [nnp_file]
        opt_nnp_file = os.path.join(
            os.path.dirname(nnp_file), f"{case_name}-opt.nnp")
        args.output_file = [opt_nnp_file]
        opt_cli.optimization_command(args)

        info_expected = load.load(nnp_file, batch_size=1)
        info_actual = load.load(opt_nnp_file, batch_size=1)

        for expected, actual in zip(forward(info_expected), forward(info_actual)):
            assert np.allclose(expected, actual)
