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

from nnabla.config import nnabla_config
from nnabla.logger import logger

################################################################################
# Import callback module
callback = None

import importlib
try:
    callback = importlib.import_module(nnabla_config.get('CALLBACK',
                                                         'util_callback_module'))
except:
    callback = None


def alternative_cli(args):
    if callback is not None and 'alternative_cli' in dir(callback):
        return callback.alternative_cli(args)
    else:
        return None


def get_callback_version():
    if callback is not None:
        return callback.get_callback_version()
    else:
        return None


def get_best_from_status(args):
    if callback is not None and 'get_best_from_status' in dir(callback):
        return callback.get_best_from_status(args)
    else:
        return None, None


def update_time_train(prediction=None):
    if callback is not None and 'update_time_train' in dir(callback):
        callback.update_time_train(prediction)


def save_train_snapshot():
    if callback is not None and 'save_train_snapshot' in dir(callback):
        callback.save_train_snapshot()


def update_status(state=None, start=False, start_time=None):
    if callback is not None and 'update_status' in dir(callback):
        callback.update_status(state, start, start_time)


def add_train_command_arg(subparser):
    if callback is not None and 'add_train_command_arg' in dir(callback):
        callback.add_train_command_arg(subparser)


def get_timelimit(args):
    if callback is not None and 'get_timelimit' in dir(callback):
        return callback.get_timelimit(args)
    else:
        return -1


def process_evaluation_result(outdir, filename):
    if callback is not None and 'process_evaluation_result' in dir(callback):
        callback.process_evaluation_result(outdir, filename)


def update_forward_time():
    if callback is not None and 'update_forward_time' in dir(callback):
        callback.update_forward_time()


def check_training_time(args, config, timeinfo, epoch, last_epoch):
    if callback is not None and 'check_training_time' in dir(callback):
        return callback.check_training_time(args, config, timeinfo, epoch, last_epoch)
    else:
        return True


def result_base(base, suffix, outdir):
    if callback is not None and 'result_base' in dir(callback):
        return callback.result_base(base, suffix, outdir)
    else:
        return None


def update_progress(text):
    if callback is not None and 'update_progress' in dir(callback):
        callback.update_progress(text)


def get_load_image_func(ext):
    if callback is not None and 'get_load_image_func' in dir(callback):
        return callback.get_load_image_func(ext)
    else:
        return None
