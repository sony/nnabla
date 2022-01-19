# Copyright 2019,2020,2021 Sony Corporation.
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

from nnabla.config import nnabla_config

################################################################################
# Import callback module
callback_list = []

import importlib
try:
    module_list_str = nnabla_config.get('CALLBACK', 'util_callback_module')
    module_list = module_list_str.strip('[]').replace(' ', '').split(',')
    for module in module_list:
        callback_list.append(importlib.import_module(module))
except:
    callback_list = []


def _get_callback(func_name):
    for callback in callback_list:
        if func_name in dir(callback):
            return callback
    return None


def alternative_cli(args):
    callback = _get_callback("alternative_cli")
    if callback:
        return callback.alternative_cli(args)
    else:
        return None


def get_callback_version():
    callback = _get_callback("get_callback_version")
    if callback:
        return callback.get_callback_version()
    else:
        return None


def get_best_from_status(args):
    callback = _get_callback("get_best_from_status")
    if callback:
        return callback.get_best_from_status(args)
    else:
        return None, None


def update_time_train(prediction=None):
    callback = _get_callback("update_time_train")
    if callback:
        callback.update_time_train(prediction)


def save_train_snapshot():
    callback = _get_callback("save_train_snapshot")
    if callback:
        callback.save_train_snapshot()


def update_status(state=None, start=False, start_time=None):
    callback = _get_callback("update_status")
    if callback:
        callback.update_status(state, start, start_time)


def add_train_command_arg(subparser):
    callback = _get_callback('add_train_command_arg')
    if callback:
        callback.add_train_command_arg(subparser)


def get_timelimit(args):
    callback = _get_callback("get_timelimit")
    if callback:
        return callback.get_timelimit(args)
    else:
        return -1


def process_evaluation_result(outdir, filename):
    callback = _get_callback("process_evaluation_result")
    if callback:
        callback.process_evaluation_result(outdir, filename)


def update_forward_time():
    callback = _get_callback("update_forward_time")
    if callback:
        callback.update_forward_time()


def check_training_time(args, config, timeinfo, epoch, last_epoch):
    callback = _get_callback("check_training_time")
    if callback:
        return callback.check_training_time(args, config, timeinfo, epoch, last_epoch)
    else:
        return True


def result_base(base, suffix, outdir):
    callback = _get_callback("result_base")
    if callback:
        return callback.result_base(base, suffix, outdir)
    else:
        return None


def update_progress(text):
    callback = _get_callback("update_progress")
    if callback:
        callback.update_progress(text)


def get_load_image_func(ext):
    callback = _get_callback("get_load_image_func")
    if callback:
        return callback.get_load_image_func(ext)
    else:
        return None
