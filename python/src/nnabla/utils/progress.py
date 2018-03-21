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
from nnabla.config import nnabla_config

# state output
# ============
state_file_name = ''
last_state_datetime = datetime.now()
last_progress = 0
log_display_progress = (nnabla_config.get(
    'LOG', 'log_display_progress') == 'True')


def configure_progress(file_name):
    global state_file_name
    state_file_name = file_name
    progress(None)


def progress(state, progress=0.0):

    global last_state_datetime
    time_to_update_progress = False
    if last_state_datetime < datetime.now() + timedelta(milliseconds=-1000) or state is None:
        last_state_datetime = datetime.now()
        time_to_update_progress = True

    if state is not None:
        global log_display_progress
        if log_display_progress:
            global last_progress
            if progress < last_progress or progress - last_progress > 0.1 or time_to_update_progress:
                logger.log(99, state + ' ({0:3.2f}%)'.format(progress * 100))
                last_progress = progress

        if time_to_update_progress:
            if len(state_file_name):
                with open(state_file_name, 'w') as f:
                    f.write(state + ' ({0:3.2f}%)'.format(progress * 100))
