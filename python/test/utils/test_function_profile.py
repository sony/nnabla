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

import time

import pytest

import nnabla.utils.function_profile as fp


def wait_one():
    time.sleep(0.1)


def wait_two():
    time.sleep(0.2)


def cond(a, b, c=None, d=None):
    return c is not None


def foo(a, b, c=None, d=None):
    for i in range(3):
        wait_one()
    for j in range(4):
        wait_two()


def test_function_profile():
    bar = fp.profile(foo)
    for i in range(3):
        bar(1, 2)
    bar.profiler.print_stats(reset=False)
    bar.profiler.print_stats(reset=True)
    bar = fp.profile(foo)
    for i in range(3):
        bar(1, 2)


@pytest.mark.parametrize("condition", [cond])
@pytest.mark.parametrize("print_freq", [2])
def test_function_profile_options(condition, print_freq):
    bar = fp.profile(condition=condition, print_freq=print_freq)(foo)
    for i in range(3):
        bar(1, 2)
    bar.profiler.print_stats(reset=False)
    bar.profiler.print_stats(reset=True)
    bar = fp.profile(foo)
    for i in range(3):
        bar(1, 2)
