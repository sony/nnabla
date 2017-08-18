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

'''Saving classification network NNP file.

This script converts HDF5 model parameter file trained with classification.py
into NNP file, a file format of NNabla, with a network definition. Before
running this script, run classification.py with arbitrary command line options.
 This script must be run with the same command line options as that used at
the training script.
'''

# Python 2/3
from __future__ import absolute_import, print_function
from six.moves import range

# Local modules
from args import get_args
from classification import mnist_lenet_prediction, mnist_resnet_prediction

# Python packages
import nnabla as nn
import nnabla.logger as logger
import nnabla.utils.save
import os


def main():
    args = get_args(description=__doc__)

    mnist_cnn_prediction = mnist_lenet_prediction
    if args.net == 'resnet':
        mnist_cnn_prediction = mnist_resnet_prediction

    # Infer parameter file name and read it.
    parameter_file = os.path.join(
        args.model_save_path,
        '{}_params_{:06}.h5'.format(args.net, args.max_iter))
    try:
        _ = nn.load_parameters(parameter_file)
    except IOError:
        logger.error("Run classification.py before runnning this script.")
        exit(1)

    # Create a computation graph to be saved.
    image = nn.Variable([args.batch_size, 1, 28, 28])
    pred = mnist_cnn_prediction(image, test=True)

    # Save NNP file (used in C++ inference later.).
    nnp_file = os.path.join(
        args.model_save_path, '{}_{:06}.nnp'.format(args.net, args.max_iter))
    runtime_contents = {
        'networks': [
            {'name': 'runtime',
             'batch_size': args.batch_size,
             'outputs': {'y': pred},
             'names': {'x': image}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'data': ['x'],
             'output': ['y']}]}
    nn.utils.save.save(nnp_file, runtime_contents)


if __name__ == '__main__':
    main()
