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


import pytest
from nnabla import ext_utils
ext_names = ext_utils.list_extensions()


@pytest.mark.parametrize('ext_name', ext_names)
def test_import_extension_module(ext_name):
    ext = ext_utils.import_extension_module(ext_name)


@pytest.mark.parametrize('ext_name', ext_names)
def test_get_extension_context(ext_name):
    ctx = ext_utils.get_extension_context(ext_name)


@pytest.mark.parametrize('ext_name', ext_names)
def test_ext_utils_misc(ext_name):
    ext = ext_utils.import_extension_module(ext_name)
    ext.clear_memory_cache()
    if ext.get_device_count() == 0:
        return
    ds = ext.get_devices()
    print(ds)
    ext.device_synchronize(ds[0])
