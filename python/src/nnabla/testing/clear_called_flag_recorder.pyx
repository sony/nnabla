# Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

from clear_called_flag_recorder cimport *

def activate_clear_called_flag_recorder():
    c_activate_clear_called_flag_recorder()

def deactivate_clear_called_flag_recorder():
    c_deactivate_clear_called_flag_recorder()

def get_input_clear_called_flags():
    return c_get_input_clear_called_flags()

def get_output_clear_called_flags():
    return c_get_output_clear_called_flags()