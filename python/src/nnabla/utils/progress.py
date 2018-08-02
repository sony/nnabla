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

from datetime import datetime, timedelta


# state output
# ============
state_file_name = ''
last_state_datetime = datetime.now()

# callback
# ============
state_callback = None


def configure_progress(file_name, callback=None):
    if file_name is not None:
        global state_file_name
        state_file_name = file_name
    if callback is not None:
        global state_callback
        state_callback = callback
    progress(None)


def progress(state, progress=0.0):
    if len(state_file_name):
        global last_state_datetime
        if last_state_datetime < datetime.now() + timedelta(milliseconds=-1000) or state is None:
            last_state_datetime = datetime.now()
            with open(state_file_name, 'w') as f:
                if state is not None:
                    f.write(state + ' ({0:3.2f}%)'.format(progress * 100))
    if state_callback is not None:
        state_callback(state, progress)
