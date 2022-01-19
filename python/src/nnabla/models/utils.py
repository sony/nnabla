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
from __future__ import absolute_import

import os

from nnabla import logger
from nnabla.utils.download import get_data_home


def get_model_home():
    '''
    Returns a root folder path for downloading models.
    '''
    d = os.path.join(get_data_home(), 'nnp_models')
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def get_model_url_base_from_env():
    '''
    Returns a value of environment variable ``NNABLA_MODELS_URL_BASE`` if set,
    otherwise returns ``None``.
    '''
    return os.environ.get('NNABLA_MODELS_URL_BASE', None)


def get_model_url_base():
    '''
    Returns a root folder for models.
    '''
    url_base = get_model_url_base_from_env()
    if url_base is not None:
        logger.info('NNBLA_MODELS_URL_BASE is set as {}.'.format(url_base))
    else:
        url_base = 'https://nnabla.org/pretrained-models/nnp_models/'
    return url_base
