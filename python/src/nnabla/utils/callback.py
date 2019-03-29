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


def get_callback_version():
    if callback is not None:
        logger.debug('get_callback_version use callback')
        return callback.get_callback_version()
    else:
        logger.debug('get_callback_version do not use callback')
        return None


def get_best_from_status(args):
    if callback is not None:
        logger.debug('get_best_from_status use callback')
        return callback.get_best_from_status(args)
    else:
        logger.debug('get_best_from_status do not use callback')
        return None, None


def update_time_train(prediction=None):
    if callback is not None:
        logger.debug('get_best_from_status use callback')
        callback.update_time_train(prediction)
    else:
        logger.debug('get_best_from_status do not use callback')


def save_train_snapshot():
    if callback is not None:
        logger.debug('save_train_snapshot use callback')
        callback.save_train_snapshot()
    else:
        logger.debug('save_train_snapshot do not use callback')


def update_status(state=None, start=False, start_time=None):
    if callback is not None:
        logger.debug('update_status use callback')
        callback.update_status(state, start, start_time)
    else:
        logger.debug('update_status do not use callback')


def add_train_command_arg(subparser):
    if callback is not None:
        logger.debug('add_train_command_arg use callback')
        callback.add_train_command_arg(subparser)
    else:
        logger.debug('add_train_command_arg do not use callback')


def get_timelimit(args):
    if callback is not None:
        logger.debug('get_timelimit use callback')
        return callback.get_timelimit(args)
    else:
        logger.debug('get_timelimit do not use callback')
        return -1


def process_evaluation_result(args, row0, rows):
    if callback is not None:
        logger.debug('process_evaluation_result use callback')
        callback.process_evaluation_result(args, row0, rows)
    else:
        logger.debug('process_evaluation_result do not use callback')


def update_forward_result(header, rows):
    if callback is not None:
        logger.debug('update_forward_result use callback')
        callback.update_forward_result(header, rows)
    else:
        logger.debug('update_forward_result do not use callback')


def update_forward_time():
    if callback is not None:
        logger.debug('update_forward_time use callback')
        callback.update_forward_time()
    else:
        logger.debug('update_forward_time do not use callback')


def check_training_time(args, config, timeinfo, epoch, last_epoch):
    if callback is not None:
        logger.debug('check_training_time use callback')
        return callback.check_training_time(args, config, timeinfo, epoch, last_epoch)
    else:
        logger.debug('check_training_time do not use callback')
        return True
