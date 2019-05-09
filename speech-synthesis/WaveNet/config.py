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


class WavenetConfig(object):
    hidden_dims = 32
    skip_dims = 512
    kernel_size = 2
    speaker_dims = 32
    dilations = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    ]


class DataConfig(object):
    sample_rate = 16000
    duration = 16000
    shift = 8000
    q_bit_len = 256


wavenet_config = WavenetConfig()
data_config = DataConfig()
