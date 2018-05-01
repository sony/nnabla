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
Parser utilities for Darknet weight files,
used inside functions in darknet19.py or yolov2.py.
'''


def get_convolutional_params(params, prefix, no_bn=False):
    '''
    Returns conv/W, bn/beta, bn/gamma, bn/mean, bn/var

    '''
    if no_bn:
        return (
            params['/'.join((prefix, 'conv/W'))],
            params['/'.join((prefix, 'conv/b'))],
        )
    return (
        params['/'.join((prefix, 'conv/W'))],
        params['/'.join((prefix, 'bn/beta'))],
        params['/'.join((prefix, 'bn/gamma'))],
        params['/'.join((prefix, 'bn/mean'))],
        params['/'.join((prefix, 'bn/var'))],
    )


def set_param_and_get_next_cursor(dn_params, cursor, param):
    '''
    '''
    param.d = dn_params[cursor:cursor + param.size].reshape(param.shape)
    return cursor + param.size


def load_convolutional_and_get_next_cursor_core(dn_params, cursor, w, b,
                                                g=None, m=None, v=None):
    '''
    '''
    cursor = set_param_and_get_next_cursor(dn_params, cursor, b)
    if g is not None:
        assert (m is not None and v is not None)
        cursor = set_param_and_get_next_cursor(dn_params, cursor, g)
        cursor = set_param_and_get_next_cursor(dn_params, cursor, m)
        cursor = set_param_and_get_next_cursor(dn_params, cursor, v)
    cursor = set_param_and_get_next_cursor(dn_params, cursor, w)
    return cursor


def load_convolutional_and_get_next_cursor(dn_params, cursor,
                                           params, prefix, no_bn=False):
    '''
    '''
    return load_convolutional_and_get_next_cursor_core(
        dn_params, cursor, *get_convolutional_params(params, prefix, no_bn))
