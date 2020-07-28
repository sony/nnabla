# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
import numpy as np

import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla as nn
import nnabla.communicators as C

from configparser import ConfigParser
from collections import defaultdict

from nnabla.ext_utils import get_extension_context
from nnabla.testing import assert_allclose


# -------
# Testers
# -------


def structure_tester(v_ref, v_act):
    ginfo_ref = GraphInfo(v_ref)
    ginfo_act = GraphInfo(v_act)

    print('====================')
    print(ginfo_ref.funcs)
    print('====================')
    print(ginfo_act.funcs)
    print('====================')

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


def value_tester(v_ref, v_act, rtol=1e-04, atol=1e-05):
    from nbla_test_utils import ArrayDiffStats

    v_ref.forward()
    v_act.forward()
    print(ArrayDiffStats(v_ref.d, v_act.d))
    assert_allclose(v_ref.d, v_act.d, rtol=rtol, atol=atol)


# -------
# Helpers
# -------


class GraphInfo(object):
    class Functor(object):

        def __init__(self, funcs, inputs, outputs,
                     variable_to_funcs):
            self.funcs = funcs
            self.inputs = inputs
            self.outputs = outputs
            self.variable_to_funcs = variable_to_funcs

        def __call__(self, func):
            self.funcs.append(func)
            self.inputs.append(func.inputs)
            self.outputs.append(func.outputs)
            for i in func.inputs:
                self.variable_to_funcs[i].append(func)

    def __init__(self, pred):
        self.funcs = []
        self.inputs = []
        self.outputs = []
        self.variable_to_funcs = defaultdict(list)

        functor = GraphInfo.Functor(self.funcs,
                                    self.inputs, self.outputs,
                                    self.variable_to_funcs)
        pred.visit(functor)
