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

'''
Wrapper module for logging.

You can use the logger as follows:

.. code-block:: python

    from utils.logger import logger

    logger.debug('Log message(DEBUG)')
    logger.info('Log message(INFO)')
    logger.error('Log message(ERROR)')
    logger.critical('Log message(CRITICAL)')


With the default settings, it should yield the following output:

.. code-block:: shell

    $ python scripts/logger_test.py
    [nnabla][ERROR]: logger_test.py : <module> : 5 : Log message(ERROR)
    [nnabla][CRITICAL]: logger_test.py : <module> : 6 : Log message(CRITICAL)
    $ cat /tmp/nbla.log
    2017-01-19 14:41:35,132 [nnabla][DEBUG]: scripts/logger_test.py : <module> : 3 : Log message(DEBUG)
    2017-01-19 14:41:35,132 [nnabla][INFO]: scripts/logger_test.py : <module> : 4 : Log message(INFO)
    2017-01-19 14:41:35,132 [nnabla][ERROR]: scripts/logger_test.py : <module> : 5 : Log message(ERROR)
    2017-01-19 14:41:35,132 [nnabla][CRITICAL]: scripts/logger_test.py : <module> : 6 : Log message(CRITICAL)


'''

import logging
import os
from os.path import abspath, join, expanduser, exists, dirname

from .config import nnabla_config


def _string_to_level(string):
    if string == 'DEBUG':
        return logging.DEBUG
    elif string == 'INFO':
        return logging.INFO
    elif string == 'WARNING':
        return logging.WARNING
    elif string == 'ERROR':
        return logging.ERROR
    elif string == 'CRITICAL':
        return logging.CRITICAL
    return None


if nnabla_config.get('LOG', 'log_file_name') == '':
    if os.name == 'posix':
        log_file_name_base = abspath(
            join(expanduser('~'), 'nnabla_data', 'log', 'nnabla'))
    elif os.name == 'nt':
        from win32com.shell import shell, shellcon
        log_file_name_base = abspath(join(shell.SHGetFolderPath(
            0, shellcon.CSIDL_APPDATA, None, 0), 'NNabla', 'log', 'nnabla'))
    try:
        os.makedirs(dirname(log_file_name_base))
    except OSError:
        pass  # python2 does not support exists_ok arg

    suffix = '_{}.log'.format(os.getpid())
    num = 1
    while exists(log_file_name_base + suffix):
        suffix = '_{}_{}.log'.format(os.getpid(), num)
        num += 1
    log_file_name = log_file_name_base + suffix
else:
    log_file_name = nnabla_config.get('LOG', 'log_file_name')

log_console_level = _string_to_level(
    nnabla_config.get('LOG', 'log_console_level'))
log_console_format = nnabla_config.get('LOG', 'log_console_format')

log_file_level = _string_to_level(nnabla_config.get('LOG', 'log_file_level'))
log_file_format = nnabla_config.get('LOG', 'log_file_format')

# set up logging to file - see previous section for more details
logging.basicConfig(level=log_file_level,
                    format=log_file_format,
                    filename=log_file_name,
                    filemode='w+')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(log_console_level)
# set a format which is simpler for console use
formatter = logging.Formatter(log_console_format)
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger('nnabla')
