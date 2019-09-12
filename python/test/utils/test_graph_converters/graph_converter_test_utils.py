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


import numpy as np

import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla as nn
import nnabla.communicators as C
from nnabla.ext_utils import get_extension_context

import nnabla.experimental.graph_converters as GC

from nnabla.testing import assert_allclose

# -------
# Testers
# -------


def structure_tester(vleaf_ref, vleaf_act):
    ginfo_ref = GC.GraphInfo(vleaf_ref)
    ginfo_act = GC.GraphInfo(vleaf_act)

    # Length check
    assert len(ginfo_ref.funcs) == len(ginfo_act.funcs)
    assert len(ginfo_ref.inputs) == len(ginfo_act.inputs)
    assert len(ginfo_ref.outputs) == len(ginfo_act.outputs)

    # Input shape check
    for is_ref, is_act in zip(ginfo_ref.inputs, ginfo_act.inputs):
        assert len(is_ref) == len(is_act)
        for i_ref, i_act in zip(is_ref, is_act):
            assert i_ref.shape == i_act.shape

    # Output shape check
    for os_ref, os_act in zip(ginfo_ref.outputs, ginfo_act.outputs):
        assert len(os_ref) == len(os_act)
        for o_ref, o_act in zip(os_ref, os_act):
            assert o_ref.shape == o_act.shape

    # Func name check
    for f_ref, f_act in zip(ginfo_ref.funcs, ginfo_act.funcs):
        assert f_ref.name == f_act.name


def value_tester(vleaf_ref, vleaf_act, rtol=1e-04, atol=1e-05):
    vleaf_ref.forward()
    vleaf_act.forward()
    print("--- Values----")
    print(vleaf_ref.d, vleaf_act.d)
    print("--- Abs Diff ----")
    print(np.abs(vleaf_ref.d - vleaf_act.d))
    print("--- Sign Diff ----")
    print(np.sign(vleaf_ref.d) - np.sign(vleaf_act.d))
    assert_allclose(vleaf_ref.d, vleaf_act.d, rtol=rtol, atol=atol)

# -------
# Helpers
# -------


def set_same_params(params_ref, params_act):
    for i0, i1 in zip(params_ref.items(), params_act.items()):
        k0, v0 = i0
        k1, v1 = i1
        v0.d = v1.d.copy()


def print_params(params_ref, params_act, only_diff=False):
    for i0, i1 in zip(params_ref.items(), params_act.items()):
        k0, v0 = i0
        k1, v1 = i1
        print("params_ref, params_act", k0, k1)
        print("params_ref, params_act", v0.shape, v1.shape)
