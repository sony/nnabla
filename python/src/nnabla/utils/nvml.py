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

import os
import sys
import pynvml


def load_nvml_for_win():
    """
    For the bug NVML Shared Library Not Found on windows, because the file nvml.dll maybe
    in the `C:/Program Files/NVIDIA Corporation/NVSMI` or the `C:/Windows/System32`.
    """
    from pynvml import nvml
    if not (nvml.nvml_lib == None and sys.platform[:3] == "win"):
        return None

    nvml.lib_load_lock.acquire()
    nvml_path = os.path.join(os.getenv(
        "ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/nvml.dll")
    if not os.path.exists(nvml_path):
        nvml_path = os.path.join("C:/windows", "system32", "nvml.dll")
    if os.path.exists(nvml_path):
        nvml.nvml_lib = nvml.CDLL(nvml_path)
    nvml.lib_load_lock.release()


load_nvml_for_win()
