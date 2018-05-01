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

from darknet19 import conv_bn_pool, darknet19_feature


def reorg_ref(x, stride):
    import numpy as np
    batch, out_c, h, w = x.shape
    c = int(out_c * stride * stride)
    h = int(h / stride)
    w = int(w / stride)
    y = np.zeros_like(x).flatten()
    x = x.flatten()
    out_c = int(c / (stride * stride))
    for i in range(x.size):
        in_index = i
        in_w = i % w
        i = int(i / w)
        in_h = i % h
        i = int(i / h)
        in_c = i % c
        i = int(i / c)
        b = i % batch
        c2 = in_c % out_c
        offset = int(in_c / out_c)
        w2 = in_w * stride + offset % stride
        h2 = in_h * stride + int(offset / stride)
        out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b))
        y[in_index] = x[out_index]
    return y.reshape(batch, c, h, w)


def reorg_ref_darknet(x, stride):
    import numpy as np
    batch, c, h, w = x.shape
    y = np.zeros_like(x).flatten()
    x = x.flatten()
    out_c = int(c / (stride * stride))
    for i in range(x.size):
        in_index = i
        in_w = i % w
        i = int(i / w)
        in_h = i % h
        i = int(i / h)
        in_c = i % c
        i = int(i / c)
        b = i % batch
        c2 = in_c % out_c
        offset = int(in_c / out_c)
        w2 = in_w * stride + offset % stride
        h2 = in_h * stride + int(offset / stride)
        out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b))
        y[in_index] = x[out_index]
    return y.reshape(batch, c * stride * stride, int(h / stride), int(w / stride))


def reorg(x, stride):
    # Input shape
    b, c, h, w = x.shape
    # Output shape
    assert h % stride == 0
    assert w % stride == 0
    C, H, W = stride * stride * c, int(h / stride), int(w / stride)
    # Reorg opration in Darknet can be done by transpose and reshape.
    r = F.reshape(x, (b, c, H, stride, W, stride))
    r = F.transpose(r, (0, 3, 5, 1, 2, 4))
    r = F.reshape(r, (b, C, H, W))
    # Note that an actual computation is only fired at the transpose
    # because `reshape` doesn't involve any computation.
    return r


def reorg_darknet_bug(x, stride):
    '''Simulate DarkNet's reorg layer including an indexing bug.
    '''
    b, c, h, w = x.shape
    assert h % stride == 0
    assert w % stride == 0
    c_bug, h_bug, h_bug = int(c / stride / stride), h * stride, w * stride
    C_bug, H_bug, W_bug = c, h, w
    C, H, W = stride * stride * c, int(h / stride), int(w / stride)
    r = F.reshape(x, (b, c_bug, h, stride, w, stride))
    r = F.transpose(r, (0, 3, 5, 1, 2, 4))
    r = F.reshape(r, (b, C, H, W))
    return r


def yolov2_feature(c13, c18, test=False, feature_dict=None):
    '''
    '''
    if feature_dict is None:
        feature_dict = {}
    # Extra feature extraction for c18
    h = conv_bn_pool(c18, 1024, 3, pool=False, test=test, name='c18_19')
    feature_dict['c18_19'] = h
    h = conv_bn_pool(h, 1024, 3, pool=False, test=test, name='c18_20')
    feature_dict['c18_20'] = h

    # Extra feature extraction for c13
    c13_h = conv_bn_pool(c13, 64, 1, pool=False, test=test, name='c13_14')
    feature_dict['c13_14'] = c13_h
    c13_h = reorg_darknet_bug(c13_h, 2)
    feature_dict['reorg'] = c13_h

    # Concatenate c13 and c18 features together
    h = F.concatenate(c13_h, h, axis=1)
    feature_dict['route'] = h

    # Extra feature extraction of the multi-scale features
    h = conv_bn_pool(h, 1024, 3, pool=False, test=test, name='c21')
    feature_dict['c21'] = h
    return h


def yolov2_detection_layer(h, num_anchors, num_classes):
    '''
    '''
    return PF.convolution(h, (4 + 1 + num_classes) *
                          num_anchors, (1, 1), name='detection')


def yolov2(x, num_anchors, num_classes, test=False, feature_dict=None):
    feature_dict_d = {}
    c18 = darknet19_feature(x, test=test, feature_dict=feature_dict_d)
    c13 = feature_dict_d['c13']
    c21 = yolov2_feature(c13, c18, test=test, feature_dict=feature_dict)
    det = yolov2_detection_layer(c21, num_anchors, num_classes)
    if feature_dict is not None:
        feature_dict.update(feature_dict_d)
        feature_dict['c21'] = c21
        feature_dict['det'] = det
    return det


def yolov2_activate(x, anchors, biases):
    shape = x.shape
    y = F.reshape(x, (shape[0], anchors, -1,) + shape[2:])
    stop = list(y.shape)
    stop[2] = 2
    t_xy = F.slice(y, (0, 0, 0, 0, 0), stop)
    stop[2] = 4
    t_wh = F.slice(y, (0, 0, 2, 0, 0), stop)
    stop[2] = 5
    t_o = F.slice(y, (0, 0, 4, 0, 0), stop)
    stop[2] = y.shape[2]
    t_p = F.slice(y, (0, 0, 5, 0, 0), stop)
    t_xy = F.sigmoid(t_xy)
    t_wh = F.exp(t_wh)
    t_o = F.sigmoid(t_o)
    t_p = F.softmax(t_p, axis=2)
    t_x, t_y, t_wh = yolov2_image_coordinate(t_xy, t_wh, biases)
    y = F.concatenate(t_x, t_y, t_wh, t_o, t_p, axis=2)
    y = F.transpose(y, (0, 1, 3, 4, 2)).reshape((
        shape[0], -1, shape[1] / anchors))
    return y


def yolov2_image_coordinate(t_xy, t_wh, biases):
    import numpy as np
    from nnabla.parameter import pop_parameter, set_parameter
    h, w = t_xy.shape[-2:]
    xs = pop_parameter('xs')
    ys = pop_parameter('ys')
    if xs is None or (h != xs.shape[-1]):
        xs = nn.Variable.from_numpy_array(np.arange(w).reshape(1, 1, 1, -1))
        xs.need_grad = False
        set_parameter('xs', xs)
    if ys is None or (h != ys.shape[-2]):
        ys = nn.Variable.from_numpy_array(np.arange(h).reshape(1, 1, -1, 1))
        ys.need_grad = False
        set_parameter('ys', ys)
    t_x, t_y = F.split(t_xy, axis=2)
    oshape = list(t_x.shape)
    oshape.insert(2, 1)
    t_x = F.reshape((t_x + xs) / w, oshape)
    t_y = F.reshape((t_y + ys) / h, oshape)
    pop_parameter('biases')
    biases = biases.reshape(
        1, biases.shape[0], biases.shape[1], 1, 1) / np.array([w, h]).reshape(1, 1, 2, 1, 1)
    b = nn.Variable.from_numpy_array(biases)
    b.need_grad = False
    set_parameter('biases', b)
    t_wh = t_wh * b
    return t_x, t_y, t_wh
