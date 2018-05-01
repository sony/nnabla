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


import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def conv_bn_pool(x, channels, kernel=3, act=F.leaky_relu, pool=True,
                 test=False, name='cbp'):
    '''
    No doc.

    '''
    pad = 1 if kernel == 3 else 0
    with nn.parameter_scope(name):
        h = PF.convolution(x, channels, (kernel, kernel),
                           pad=(pad, pad), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
    h = act(h)
    if pool:
        h = F.max_pooling(h, (2, 2), (2, 2))
    return h


def darknet19_feature(x, act=F.leaky_relu, test=False, feature_dict=None):
    '''

    Args:
        feature_dict (dict): If a dict given, the all variables of feature
            maps before max pooling are pushed into the dict.

    '''
    # Layer configurations are composed of
    # a list of (filters, kernel size, pooling or not)
    layer_configs = [
        # Unit1 (2)
        (32, 3, True),
        (64, 3, True),
        # Unit2 (3)
        (128, 3, False),
        (64, 1, False),
        (128, 3, True),
        # Unit3 (3)
        (256, 3, False),
        (128, 1, False),
        (256, 3, True),
        # Unit4 (5)
        (512, 3, False),
        (256, 1, False),
        (512, 3, False),
        (256, 1, False),
        (512, 3, True),
        # Unit5 (5)
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
    ]
    h = x
    for i, config in enumerate(layer_configs):
        c, k, p = config
        name = 'c{}'.format(i + 1)
        h = conv_bn_pool(h, c, k, act=act, pool=p,
                         name=name, test=test)
        if feature_dict is not None:
            if h.parent.name.startswith('MaxPooling'):
                feature_dict[name] = h.parent.inputs[0]
    return h


def darknet19_classification(x, num_class=1000, act=F.leaky_relu, test=False):
    '''
    '''
    h = darknet19_feature(x, act, test=test)
    h = PF.convolution(h, num_class, (1, 1), name='c19')
    h = F.mean(h, axis=(2, 3))
    return h


# Parser utilities of Darknet19
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
