# Copyright 2021 Sony Group Corporation.
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
from helper import create_temp_with_dir
from nnabla.utils.get_file_handle import get_file_handle_save, get_file_handle_load


@pytest.mark.parametrize("extension", [".nntxt", ".protobuf"])
def test_file_close_exception(extension):
    with pytest.raises(ZeroDivisionError) as excinfo:
        with create_temp_with_dir("tmp{}".format(extension)) as filename:
            with get_file_handle_save(filename, ext=extension) as f:
                file_handler = f
                1 / 0
    assert file_handler.closed

    with pytest.raises(ZeroDivisionError) as excinfo:
        with create_temp_with_dir("tmp{}".format(extension)) as filename:
            # create a file at first
            with open(filename, "w") as f:
                f.write("\n")

            with get_file_handle_load(None, filename, ext=extension) as f:
                file_handler = f
                1 / 0
    assert file_handler.closed
