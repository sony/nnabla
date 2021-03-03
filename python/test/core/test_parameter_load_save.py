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
import io
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from collections import OrderedDict
from nnabla.testing import assert_allclose
from nnabla.core.modules import ConvBn, ResUnit

from helper import ModuleCreator, forward_variable_and_check_equal

nnp_file = "t.nnp"
protobuf_file = "t.protobuf"
h5_file = "t.h5"

temp_files = [
    nnp_file,
    protobuf_file,
    h5_file
]


class TSTNetNormal(nn.Module):
    def __init__(self):
        self.conv_bn_1 = ConvBn(1)
        self.conv_bn_2 = ConvBn(1)

    def call(self, x1, x2):
        y1 = self.conv_bn_1(x1)
        y2 = self.conv_bn_2(x2)
        y = F.concatenate(y1, y2, axis=1)
        return y


@pytest.fixture(scope="function", autouse=True)
def fixture_for_temp_file():
    yield
    for fn in temp_files:
        if os.path.exists(fn):
            os.remove(fn)


def test_parameter_file_load_save_using_global():
    module_creator = ModuleCreator(
        TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])
    proto_variable_inputs = module_creator.get_proto_variable_inputs()
    outputs = module_creator.module(*proto_variable_inputs)
    g = nn.graph_def.get_default_graph_by_variable(outputs)
    g.save(nnp_file)
    another = TSTNetNormal()
    variable_inputs = module_creator.get_variable_inputs()
    outputs = g(*variable_inputs)
    ref_outputs = another(*variable_inputs)

    # Should not equal
    with pytest.raises(AssertionError) as excinfo:
        forward_variable_and_check_equal(outputs, ref_outputs)

    # load to global scope
    nn.load_parameters(nnp_file)
    params = nn.get_parameters()
    another.set_parameters(params)

    ref_outputs = another(*variable_inputs)
    forward_variable_and_check_equal(outputs, ref_outputs)


def test_parameter_file_load_save():
    module_creator = ModuleCreator(
        TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])
    proto_variable_inputs = module_creator.get_proto_variable_inputs()
    outputs = module_creator.module(*proto_variable_inputs)
    g = nn.graph_def.get_default_graph_by_variable(outputs)
    g.save(nnp_file)
    another = TSTNetNormal()
    variable_inputs = module_creator.get_variable_inputs()
    outputs = g(*variable_inputs)
    ref_outputs = another(*variable_inputs)

    # Should not equal
    with pytest.raises(AssertionError) as excinfo:
        forward_variable_and_check_equal(outputs, ref_outputs)

    # load to local scope
    with nn.parameter_scope('', another.parameter_scope):
        nn.load_parameters(nnp_file)

    another.update_parameter()

    ref_outputs = another(*variable_inputs)
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("parameter_file", [protobuf_file, h5_file])
def test_parameter_file_load_save_for_files(parameter_file):
    module_creator = ModuleCreator(
        TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])
    variable_inputs = module_creator.get_variable_inputs()
    a_module = module_creator.module
    outputs = a_module(*variable_inputs)
    another = TSTNetNormal()
    ref_outputs = another(*variable_inputs)

    # Should not equal
    with pytest.raises(AssertionError) as excinfo:
        forward_variable_and_check_equal(outputs, ref_outputs)

    # save to file
    nn.save_parameters(parameter_file, a_module.get_parameters())

    # load from file
    with nn.parameter_scope('', another.parameter_scope):
        nn.load_parameters(parameter_file)
    another.update_parameter()

    ref_outputs = another(*variable_inputs)

    # should equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("memory_buffer_format", ['.protobuf', '.h5'])
def test_parameter_file_load_save_for_file_object(memory_buffer_format):
    module_creator = ModuleCreator(
        TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])
    variable_inputs = module_creator.get_variable_inputs()
    a_module = module_creator.module
    outputs = a_module(*variable_inputs)
    another = TSTNetNormal()
    ref_outputs = another(*variable_inputs)
    extension = memory_buffer_format

    # Should not equal
    with pytest.raises(AssertionError) as excinfo:
        forward_variable_and_check_equal(outputs, ref_outputs)

    with io.BytesIO() as param_file:
        nn.save_parameters(
            param_file, a_module.get_parameters(), extension=extension)
        # load from file
        with nn.parameter_scope('', another.parameter_scope):
            nn.load_parameters(param_file, extension=extension)
        another.update_parameter()

    ref_outputs = another(*variable_inputs)

    # should equal
    forward_variable_and_check_equal(outputs, ref_outputs)
