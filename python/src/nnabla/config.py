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

'''
Search config file and get config information from config file.

Config file search order is described in following table.
Each config value is overwritten by the following configs.

+-------------+--------------------+-------------------------------------------------------------------------+
| Type        | Posix              | Windows                                                                 |
+=============+====================+=========================================================================+
| System wide | /etc/nnabla.conf   | c:\\\\ProgramData\\\\NNabla\\\\nnabla.ini                                     |
+-------------+--------------------+-------------------------------------------------------------------------+
| User        | ~/.nnabla          | c:\\\\Users\\\\[USERNAME]\\\\AppData\\\\Roaming\\\\NNabla\\\\nnabla.ini             |
+-------------+--------------------+-------------------------------------------------------------------------+
| Default     | (Same directory with 'config.py')/nnabla.conf                                                |
+-------------+--------------------+-------------------------------------------------------------------------+
| Local       | [CURRENT DIRECTORY]/nnabla.conf                                                              |
+-------------+----------------------------------------------------------------------------------------------+

You can get config value as followings.

.. code-block:: python

    from utils.config import nnabla_config
    value = nnabla_config.get(CATEGORY, VALUE_NAME)

CATEGORY and VALUE_NAME does not defined in config.py.
You can add CATEGORY and VALUE as you like.
See `Official document <http://docs.python.jp/3.6/library/configparser.html#mapping-protocol-access>`_ for more information.

.. code-block:: ini

    [CATEGORY]
    VALUE_NAME = value


Default values defined in 'nnabla.conf' placed same directory with config.py is here.

.. literalinclude:: ../../python/src/nnabla/nnabla.conf
   :language: ini
   :linenos:
'''


import six.moves.configparser as configparser
import os
from os.path import abspath, dirname, exists, expanduser, join


def _get_nnabla_config():
    config_files = []
    config_files.append(join(dirname(abspath(__file__)), 'nnabla.conf'))
    if os.name == 'posix':
        config_files.append('/etc/nnabla.conf')
        config_files.append(abspath(join(expanduser('~'), '.nnabla')))
        config_files.append(abspath(join(os.getcwd(), 'nnabla.conf')))
    elif os.name == 'nt':
        from win32com.shell import shell, shellcon
        config_files.append(abspath(join(shell.SHGetFolderPath(
            0, shellcon.CSIDL_COMMON_APPDATA, None, 0), 'NNabla', 'nnabla.ini')))
        config_files.append(abspath(join(shell.SHGetFolderPath(
            0, shellcon.CSIDL_APPDATA, None, 0), 'NNabla', 'nnabla.ini')))
    config_files.append(abspath(join(os.getcwd(), 'nnabla.conf')))

    if "NNABLA_CONFIG_FILE_PATH" in os.environ:
        conf = os.environ["NNABLA_CONFIG_FILE_PATH"]
        if os.path.exists(conf):
            config_files.append(conf)

    config = configparser.RawConfigParser()
    for filename in config_files:
        # print(' Checking {}'.format(filename))
        if exists(filename):
            # print(' Read from {}'.format(filename))
            config.read(filename)
    return config


nnabla_config = _get_nnabla_config()
