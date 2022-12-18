# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

"""
Wrapper module for logging.

You can use the logger as follows:

.. code-block:: python

    from utils.logger import logger

    logger.debug('Log message(DEBUG)')
    logger.info('Log message(INFO)')
    logger.warning('Log message(WARNING)')
    logger.error('Log message(ERROR)')
    logger.critical('Log message(CRITICAL)')


With the default settings, it should yield the following output:

.. code-block:: shell

    $ python scripts/logger_test.py
    [nnabla][ERROR]: logger_test.py : <module> : 5 : Log message(ERROR)
    [nnabla][CRITICAL]: logger_test.py : <module> : 6 : Log message(CRITICAL)


If you want to output log to file.
You must create `nnabla.conf` file and put following entry.

See :py:mod:`nnabla.config` for more information about config file.

.. code-block:: ini

    [LOG]
    log_file_name = /tmp/nbla.log


After this you can get following output.

.. code-block:: shell


    $ python scripts/logger_test.py
    [nnabla][ERROR]: logger_test.py : <module> : 5 : Log message(ERROR)
    [nnabla][CRITICAL]: logger_test.py : <module> : 6 : Log message(CRITICAL)
    $ cat /tmp/nbla.log
    2017-01-19 14:41:35,132 [nnabla][DEBUG]: scripts/logger_test.py : <module> : 3 : Log message(DEBUG)
    2017-01-19 14:41:35,132 [nnabla][INFO]: scripts/logger_test.py : <module> : 4 : Log message(INFO)
    2017-01-19 14:41:35,132 [nnabla][ERROR]: scripts/logger_test.py : <module> : 5 : Log message(ERROR)
    2017-01-19 14:41:35,132 [nnabla][CRITICAL]: scripts/logger_test.py : <module> : 6 : Log message(CRITICAL)


"""

import logging

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


log_console_level = _string_to_level(
    nnabla_config.get('LOG', 'log_console_level'))
log_console_format = nnabla_config.get('LOG', 'log_console_format')

logging.basicConfig(level=log_console_level, format=log_console_format)

if nnabla_config.get('LOG', 'log_file_name') != '':
    log_file_name = nnabla_config.get('LOG', 'log_file_name')
    log_file_level = _string_to_level(
        nnabla_config.get('LOG', 'log_file_level'))
    log_file_format = nnabla_config.get('LOG', 'log_file_format')
    handler = logging.FileHandler(log_file_name)
    handler.setLevel(log_file_level)
    handler.setFormatter(logging.Formatter(log_file_format))
    logging.getLogger('').addHandler(handler)

logger = logging.getLogger('nnabla')
