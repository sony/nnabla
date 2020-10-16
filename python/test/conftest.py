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
from __future__ import print_function

import pytest

from collections import namedtuple
import os

import nnabla as nn


@pytest.fixture(scope="module", autouse=True)
def nnabla_examples_root():
    root = os.environ.get('NNABLA_EXAMPLES_ROOT', '')
    if not root:
        root = os.path.join(os.path.dirname(__file__),
                            '../../../nnabla-examples')
    NNablaExamples = namedtuple(
        'NNablaExamples', ['available', 'path'])
    return NNablaExamples(os.path.isdir(root), root)


@pytest.fixture(scope='function', autouse=True)
def scope_function():
    # turn off auto forward mode
    nn.set_auto_forward(False)

    # clear all parameters
    nn.clear_parameters()

    # keep context
    ctx = nn.get_current_context()

    # use cached array
    nn.prefer_cached_array(True)

    yield

    # restore context
    nn.set_default_context(ctx)
