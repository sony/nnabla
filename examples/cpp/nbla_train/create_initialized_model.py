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

This script creates initialized model in mnist classification examples.
and save into NNP file, a file format of NNabla, with a network definition.
This script must be run with the same command line options as that used at
the training script in nnabla-examples/mnist-collections/classification.py.
'''

# Python 2/3
from __future__ import absolute_import, print_function

# Python packages
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.solvers as S
import os
import sys


def main():

    # Read envvar `NNABLA_EXAMPLES_ROOT` to identify the path to your local
    # nnabla-examples directory.
    HERE = os.path.dirname(__file__)
    nnabla_examples_root = os.environ.get('NNABLA_EXAMPLES_ROOT', os.path.join(
        HERE, '../../../../nnabla-examples'))
    mnist_examples_root = os.path.realpath(
        os.path.join(nnabla_examples_root, 'mnist-collection'))
    sys.path.append(mnist_examples_root)
    nnabla_examples_git_url = 'https://github.com/sony/nnabla-examples'

    # Check if nnabla-examples found.
    try:
        from args import get_args
    except ImportError:
        print(
            'An envvar `NNABLA_EXAMPLES_ROOT`'
            ' which locates the local path to '
            '[nnabla-examples]({})'
            ' repository must be set correctly.'.format(
                nnabla_examples_git_url),
            file=sys.stderr)
        raise

    # Import MNIST data
    from mnist_data import data_iterator_mnist
    from classification import mnist_lenet_prediction, mnist_resnet_prediction

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_epoch", "-me", type=int, default=100)
    parser.add_argument("--iter_per_epoch", "-ipe", type=int, default=937)
    parser.add_argument("--cache_dir", "-cd", type=str, default='cache')
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "-w", type=float, default=0)
    parser.add_argument("--device-id", "-d", type=str, default='0')
    parser.add_argument("--type-config", "-t", type=str, default='float')
    parser.add_argument("--net", "-n", type=str, default='lenet')
    parser.add_argument('--context', '-c', type=str,
                        default='cpu', help="Extension modules. ex) 'cpu', 'cudnn'.")
    args = parser.parse_args()

    args_added = parser.parse_args()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    mnist_cnn_prediction = mnist_lenet_prediction
    if args.net == 'resnet':
        mnist_cnn_prediction = mnist_resnet_prediction

    # Create a computation graph to be saved.
    x = nn.Variable([args.batch_size, 1, 28, 28])
    t = nn.Variable([args.batch_size, 1])
    h_t = mnist_cnn_prediction(x, test=False, aug=False)
    loss_t = F.mean(F.softmax_cross_entropy(h_t, t))
    h_v = mnist_cnn_prediction(x, test=True, aug=False)
    loss_v = F.mean(F.softmax_cross_entropy(h_v, t))

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Save NNP file (used in C++ inference later.).
    nnp_file = '{}_initialized.nnp'.format(args.net)
    training_contents = {
        'global_config': {'default_context': ctx},
        'training_config':
            {'max_epoch': args.max_epoch,
             'iter_per_epoch': args_added.iter_per_epoch,
             'save_best': True},
        'networks': [
            {'name': 'training',
             'batch_size': args.batch_size,
             'outputs': {'loss': loss_t},
             'names': {'x': x, 'y': t, 'loss': loss_t}},
            {'name': 'validation',
             'batch_size': args.batch_size,
             'outputs': {'loss': loss_v},
             'names': {'x': x, 'y': t, 'loss': loss_v}}],
        'optimizers': [
            {'name': 'optimizer',
             'solver': solver,
             'network': 'training',
             'dataset': 'mnist_training',
             'weight_decay': 0,
             'lr_decay': 1,
             'lr_decay_interval': 1,
             'update_interval': 1}],
        'datasets': [
            {'name': 'mnist_training',
             'uri': 'MNIST_TRAINING',
             'cache_dir': args.cache_dir + '/mnist_training.cache/',
             'variables': {'x': x, 'y': t},
             'shuffle': True,
             'batch_size': args.batch_size,
             'no_image_normalization': True},
            {'name': 'mnist_validation',
             'uri': 'MNIST_VALIDATION',
             'cache_dir': args.cache_dir + '/mnist_test.cache/',
             'variables': {'x': x, 'y': t},
             'shuffle': False,
             'batch_size': args.batch_size,
             'no_image_normalization': True
             }],
        'monitors': [
            {'name': 'training_loss',
             'network': 'validation',
             'dataset': 'mnist_training'},
            {'name': 'validation_loss',
             'network': 'validation',
             'dataset': 'mnist_validation'}],
    }
    nn.utils.save.save(nnp_file, training_contents)


if __name__ == '__main__':
    main()
