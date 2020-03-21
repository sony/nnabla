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
import numpy as np
import caffe


def show_top5(prob):
    inds = np.argsort(-prob)[..., :5]
    print(inds, prob[..., inds[0]])


def crop_center_image(image, crop_shape):
    H, W = image.shape[:2]
    h, w = crop_shape
    hh, ww = (H - h) // 2, (W - w) // 2
    image = image[hh:hh+h, ww:ww+w]
    return image


def show_parameters(net):
    ctp = 0
    for k, v in net.params.items():
        ctp += sum([vv.data.size for vv in v if vv.data.size != 1])
        print(k, ':', [vv.data.shape for vv in v], ctp)
    print('total parameters:', ctp)


def load_image(path):
    from nnabla.utils.image_utils import imread
    cimg = crop_center_image(imread(path, size=(256, 256)), (224, 224))
    pimg = cimg[..., ::-1].transpose(2, 0, 1)[None]  # BGR and NCHW
    mean = np.array([104, 117, 123], dtype=np.float32).reshape(1, 3, 1, 1)
    pimg = pimg - mean
    return pimg


def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('prototxt')
    p.add_argument('caffemodel')
    p.add_argument('input')
    return p.parse_args()


def main():
    args = get_args()
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    show_parameters(net)
    img = load_image(args.input)

    net.blobs['data'].data[...] = img
    output = net.forward()
    show_top5(output['prob'])


if __name__ == '__main__':
    main()
