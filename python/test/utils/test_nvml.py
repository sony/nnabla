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
from nnabla.utils import nvml


def test_nvml():
    try:
        nvml.nvmlInit()
    except nvml.NVMLError as ne:
        try:
            import subprocess
            subprocess.check_output('nvidia-smi', shell=True)
            raise
        except:
            pytest.skip("No nvidia device")

    device_count = nvml.nvmlDeviceGetCount()
    for device_index in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)

        name = nvml.nvmlDeviceGetName(handle)
        print(f"gpu_{device_index} name: {name.decode()}")

        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu_{device_index} GPU utilization: {utilization.gpu}%')

        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        print(f'gpu_{device_index} total memory: {int(memory.total/1024**2)}MB')

    nvml.nvmlShutdown()
