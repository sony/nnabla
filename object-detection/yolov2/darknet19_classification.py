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

import darknet19

import numpy as np
import time
from nnabla.utils.image_utils import imread, imresize


def get_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--context', '-c', default='cpu', help='cpu|cudnn')
    p.add_argument('--weights', '-w', default='darknet19.h5',
                   help='NNabla parameters h5')
    p.add_argument('--input', '-i', default='dog.jpg',
                   help='path to input image')
    p.add_argument('--size', '-s', default=224, help='Image width and height')
    args = p.parse_args()
    return args


def main():
    args = get_args()
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)

    nn.load_parameters(args.weights)
    x = nn.Variable((1, 3, args.size, args.size))
    y = darknet19.darknet19_classification(x / 255, test=True)

    label_names = np.loadtxt('imagenet.shortnames.list',
                             dtype=str, delimiter=',')[:1000]

    img = imread(args.input, num_channels=3)
    img = imresize(img, (args.size, args.size))

    x.d = img.transpose(2, 0, 1).reshape(1, 3, args.size, args.size)
    y.forward(clear_buffer=True)

    # softmax
    p = F.reshape(F.mul_scalar(F.softmax(y.data), 100), (y.size,))

    # Show top-5 prediction
    inds = np.argsort(y.d.flatten())[::-1][:5]
    for i in inds:
        print('{}: {:.1f}%'.format(label_names[i], p.data[i]))

    s = time.time()
    n_time = 10
    for i in range(n_time):
        y.forward(clear_buffer=True)
    # Invoking device-to-host copy to synchronize the device (if CUDA).
    _ = y.d
    print("Processing time: {:.1f} [ms/image]".format(
        (time.time() - s) / n_time * 1000))


if __name__ == '__main__':
    main()
