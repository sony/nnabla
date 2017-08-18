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

'''Create pgm files of MNIST images.

This must be run to create input of C++ mnist_runtime example.
'''

from __future__ import print_function
from six.moves import range

import os
import sys
import numpy as np


def main():
    HERE = os.path.dirname(__file__)

    # Import MNIST data
    sys.path.append(
        os.path.realpath(os.path.join(HERE, '..', '..', 'vision', 'mnist')))
    from mnist_data import data_iterator_mnist

    # Create binary output folder
    path_bin = os.path.join(HERE, "mnist_images")
    if not os.path.isdir(path_bin):
        os.makedirs(path_bin)

    # Get MNIST testing images.
    images, labels = data_iterator_mnist(
        10000, train=False, shuffle=True).next()

    # Dump image binary files with row-major order.
    for i in range(10):
        outfile = os.path.join(path_bin, "{}.pgm".format(i))
        print("Generator a binary file of number {} to {}".format(i, outfile))
        ind = np.where(labels == i)[0][0]
        image = images[ind].copy(order='C')
        with open(outfile, 'w') as fd:
            print('P5', file=fd)
            print('# Created by nnabla mnist_runtime example.', file=fd)
            print('28 28', file=fd)
            print('255', file=fd)
            image.tofile(fd)


if __name__ == '__main__':
    main()
