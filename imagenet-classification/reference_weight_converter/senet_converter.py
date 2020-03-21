# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

from __future__ import print_function

import re
import sys
sys.path.append('..')


import numpy as np

import caffe
import nnabla as nn


def get_mean_bgr():
    mean = np.array([104, 117, 123], dtype=np.float32)
    return mean


def get_args():

    import argparse
    parser = argparse.ArgumentParser(
        description='''SENets Weight format converter (SEResNet50 & SEResNeXt50)''')

    parser.add_argument('prototxt', help='Path to .prototxt file.')
    parser.add_argument('caffemodel', help='Path to .caffemodel file.')
    parser.add_argument(
        'path_save', help='Destination path of converted .h5 weight file.')
    parser.add_argument('--resnext', '-x', default=False,
                        action='store_true', help='Use ResNeXt block.')
    parser.add_argument('--bgr', '-b', default=False, action='store_true',
                        help='Keep first input channel order as BGR. Caffe model takes BGR-order images as input. By default, it changes the order of first convolution weights so that it takes RGB-order inputs.')
    parser.add_argument('--disable-category-reordering', '-D', default=False, action='store_true',
                        help='Keep the original ordering of categories in the last classification layer. When not specified, the last fully connected output dimension will be reordered to be consistent with other imagenet models of ours.')
    return parser.parse_args()


def category_reordering_indices():
    import os
    here = os.path.dirname(__file__)
    path_label = os.path.join(here, '..', 'label_wordnetid.csv')
    path_senet_label = os.path.join(here, 'senet_synsets.txt')
    label = np.genfromtxt(path_label, dtype=str, delimiter=',', usecols=(1,))
    senet_label = np.genfromtxt(path_senet_label, dtype=str)
    lut = dict(zip(senet_label, np.arange(senet_label.size)))
    inds = [lut[l] for l in label]
    return inds


def load_caffe_net(prototxt, caffemodel):

    # Load Caffe
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    param_dims = 0
    for k, v in net.params.items():
        param_dims += sum([vv.data.size for vv in v if vv.data.size != 1])
        print(k, ':', [vv.data.shape for vv in v], param_dims)
    print('total parameters:', param_dims)
    return net, param_dims


def create_nnabla_net(resnext=False):
    # Create nnabla graph
    from models import senet
    x = nn.Variable((1, 3, 224, 224), need_grad=False)
    if resnext:
        y = senet.se_resnext50(x, 1000, test=True)
    else:
        y = senet.se_resnet50(x, 1000, test=True)
    params = nn.get_parameters(grad_only=False)
    param_dims = 0
    for k, v in params.items():
        param_dims += np.prod(v.shape)
        print(k, v.shape, param_dims)
    print('total parameters: ', param_dims)
    return (x, y), params, param_dims


def conv_bn(key):
    pname = key.split('/')[-1]
    plut = {
        'W': ('', 0),
        'b': ('', 1),
        'gamma': ('/bn/scale', 0),
        'beta': ('/bn/scale', 1),
        'mean': ('/bn', (0, 2)),
        'var': ('/bn', (1, 2)),
    }
    return plut[pname]


def lut(key):
    if key.startswith('conv1/'):
        cb, ind = conv_bn(key)
        return 'conv1/7x7_s2' + cb, ind

    if key.startswith('fc/'):
        cb, ind = conv_bn(key)
        return 'classifier' + cb, ind

    pattern = re.compile(r'res(\d)/layer(\d)')
    res_id, layer_id = map(lambda x: int(x), pattern.search(key).groups())
    prefix = 'conv{}_{}'.format(res_id + 1, layer_id)

    bname = key.split('/')[2]
    if bname == 'se':
        fc = key.split('/')[3]
        if fc == 'fc1':
            suffix = '1x1_down'
        else:
            suffix = '1x1_up'

    else:
        bname_lut = {
            'bottleneck1': '1x1_reduce',
            'bottleneck2': '3x3',
            'bottleneck3': '1x1_increase',
            'bottleneck_s': '1x1_proj'
        }
        suffix = bname_lut[bname]
    cb, ind = conv_bn(key)
    return '_'.join([prefix, suffix]) + cb, ind


def convert_weights(caffe_net, params, keep_bgr, reorder):
    for k, v in params.items():
        cname, cind = lut(k)
        scale_factor = 1
        if isinstance(cind, tuple):
            # BN
            cind, sind = cind
            inv_scale_factor = caffe_net.params[cname][sind].data.item()
            assert inv_scale_factor > 0
            scale_factor = 1 / inv_scale_factor
            print(scale_factor)
        blob = caffe_net.params[cname][cind]
        print(k, ':', (cname, cind, v.shape, tuple(blob.shape)))
        data = blob.data
        if not keep_bgr and cname == 'conv1/7x7_s2' and cind == 0:
            data = data[:, ::-1]

        # Reorder the categories of classification layer
        if reorder and k.startswith('fc/fc/'):
            indices = category_reordering_indices()
            data = data[np.asarray(indices, dtype=np.int)]

        v.d.flat = data.flat
        v.d *= scale_factor


def main():
    args = get_args()
    net, caffe_param_dims = load_caffe_net(args.prototxt, args.caffemodel)
    xy, params, params_dims = create_nnabla_net(args.resnext)
    assert caffe_param_dims == params_dims, "Number of parameter dimensions must be the same."
    convert_weights(net, params, args.bgr,
                    not args.disable_category_reordering)
    nn.save_parameters(args.path_save)


if __name__ == '__main__':
    main()
