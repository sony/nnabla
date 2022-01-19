# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import time
from datetime import datetime, timedelta

import nnabla.utils.callback as callback
from nnabla import logger
from nnabla.utils.cli.utility import cpu_load_backend_ok

# state output
# ============
state_file_name = ''
last_state_datetime = datetime.now()

# callback
# ============
state_callback = None


def configure_progress(file_name, cb=None):
    if file_name is not None:
        global state_file_name
        state_file_name = file_name
    if cb is not None:
        global state_callback
        state_callback = cb
    progress(None)


def progress(state, progress=0.0):
    if len(state_file_name):
        global last_state_datetime
        if last_state_datetime < datetime.now() + timedelta(milliseconds=-1000) or state is None:
            last_state_datetime = datetime.now()
            retry = 1
            while True:
                try:
                    with open(state_file_name, 'w') as f:
                        if state is not None:
                            f.write(
                                state + ' ({0:3.2f}%)'.format(progress * 100))
                    break
                except:
                    retry += 1
                    if retry > 100:
                        logger.critical(
                            'Failed to write to {}.'.format(state_file_name))
                        raise
                    time.sleep(0.1)
    callback.update_progress('{0} ({1:3.2f}%)'.format(state, progress * 100))
    if cpu_load_backend_ok:
        callback.update_status()
    if state_callback is not None:
        state_callback(state, progress)
