# Copyright 2019,2020,2021 Sony Corporation.
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


def assert_allclose(actual, desired, rtol=1e-5, atol=1e-6, equal_nan=True,
                    err_msg='', verbose=True):
    """A wrapper of `numpy.testing.assert_allclose`.

    Using default values for `rtol` and `atol` that are consistent with
    `numpy.allclose`.
    """
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol,
                               equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)
