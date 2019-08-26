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

from six.moves import map

import nnabla as nn


def check_cached_array_preferred(ac, prefer=True):
    c = list(map(lambda x: not (prefer ^ ('Cached' in x)), ac))
    assert c == sorted(c, reverse=True)


def test_prefer_cached_array():
    nn.reset_array_preference()
    nn.prefer_cached_array(True)
    ac2 = nn.array_classes()
    check_cached_array_preferred(ac2)

    try:
        from nnabla_ext import cuda
    except:
        cuda = None
    if cuda is not None:
        ac2 = cuda.array_classes()
        check_cached_array_preferred(ac2)

    nn.prefer_cached_array(False)
    ac2 = nn.array_classes()
    check_cached_array_preferred(ac2, False)
    if cuda is not None:
        ac2 = cuda.array_classes()
        check_cached_array_preferred(ac2, False)

    nn.prefer_cached_array(True)
