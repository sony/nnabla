# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

"""
Usage Instructions:
    5-way 1-shot omniglot:
        meta-train: python3 train_maml.py --datasource omniglot --metatrain_iterations 40000 --meta_batch_size 32 --num_shots 1 --update_lr 0.4 --num_updates 1 --logdir logs_maml/omniglot1shot5way/
        meta-test: python3 train_maml.py --datasource omniglot --meta_batch_size 32 --num_shots 1 --update_lr 0.4 --num_updates 1 --logdir logs_maml/omniglot1shot5way/ --param_file params40000.h5 --test
"""

import os
import csv
import numpy as np
import pickle
import random
import argparse
from collections import OrderedDict
from distutils.util import strtobool
import time

import nnabla as nn
import nnabla.logger as logger
import nnabla.solver as S
import nnabla.functions as F
from nnabla.monitor import Monitor, MonitorSeries

from data_generator_maml import DataGenerator
from net import net


def get_args():
    parser = argparse.ArgumentParser(description='MAML for Few-shot Learning')
    # Dataset/method options
    parser.add_argument('--dataset_root', '-dr', type=str, default='.')
    parser.add_argument('--datasource', type=str, default='omniglot',
                        help='omniglot')
    parser.add_argument('--method', type=str, default='maml',
                        help='use the authors\' network architecture of MAML')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='number of classes used in classification (N for N-way classification)')
    # Training options
    parser.add_argument('--metatrain_iterations', type=int, default=40000,
                        help='number of metatraining iterations')
    parser.add_argument('--meta_batch_size', type=int, default=32,
                        help='number of tasks sampled per meta-update')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='the base learning rate of the generator')
    parser.add_argument('--num_shots', type=int, default=1,
                        help='number of examples used for inner gradient update (K for K-shot learning)')
    parser.add_argument('--num_queries', type=int, default=-1,
                        help='number of examples per task used for meta gradient update (use if you want to inner test with a different value from num_shots)')
    parser.add_argument('--update_lr', type=float, default=0.4,
                        help='step size alpha for inner gradient update')
    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of inner gradient updates during meta training')
    parser.add_argument('--test_num_updates', type=int, default=10,
                        help='number of inner gradient updates during meta testing')
    parser.add_argument('--num_test_points', type=int, default=600,
                        help='number of tasks for meta test')
    parser.add_argument('--first_order', type=strtobool, default='False',
                        help='if True, use first-order approximation in meta-optimization')
    # Model options
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help='batch_norm or None')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='number of filters for conv nets -- 64 for omniglot')
    parser.add_argument('--max_pool', action='store_true',
                        help='Whether or not to use max pooling rather than strided convolutions (default = False)')
    # Logging, saving, and testing options
    parser.add_argument('--logdir', type=str, default='./tmp/data',
                        help='directory for checkpoints and monitor')
    parser.add_argument('--test', action='store_true',
                        help='True to train, False to test (default = False)')
    parser.add_argument('--param_file', type=str, default='params40000.h5',
                        help='filename of params to load')
    parser.add_argument('--train_num_shots', type=int, default=-1,
                        help='number of examples used for inner gradient update during meta training (use if you want to meta test with a different number)')
    parser.add_argument('--train_num_queries', type=int, default=-1,
                        help='number of examples used for meta gradient update during meta training (use if you want to meta test with a different number)')
    parser.add_argument('--train_update_lr', type=float, default=-1,
                        help='value of inner gradient step during meta training. (use if you want to meta test with a different value)')
    # Device/config options
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help='Extension modules. ex) "cpu", "cudnn"')
    parser.add_argument('--device_id', '-gid', type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`')
    parser.add_argument('--type-config', '-t', type=str, default='float',
                        help='Type of computation. e.g. "float", "half"')
    # Interval options
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--test_print_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=1000)
    args = parser.parse_args()
    return args


def inner_train_test(inputa, inputb, labela, labelb, data_generator, meta_training, args):
    lossesa, lossesb, accuraciesa, accuraciesb = [], [], [], []
    if meta_training:
        num_updates = args.num_updates
        update_lr = args.train_update_lr
    else:
        num_updates = args.test_num_updates
        update_lr = args.update_lr

    # Training
    for inp in data_generator.next():
        inputa.d, inputb.d, labela.d, labelb.d = inp

        # Initialize network
        with nn.parameter_scope('meta'):
            resulta = net(inputa, labela, True, args)
            resultb = net(inputb, labelb, True, args)
            fast_weights = nn.get_parameters()

        # For saving training accuracies
        resulta[0].persistent = True
        resulta[1].persistent = True
        task_lossa_var = [resulta[0], ]
        task_accuracya_var = [resulta[1], ]

        # Inner loop
        for j in range(num_updates):
            grad_list = nn.grad(resulta[0], fast_weights.values())
            for ind, key in enumerate(fast_weights.keys()):
                if grad_list[ind] is None:
                    continue
                if args.first_order or not meta_training:
                    grad_list[ind].need_grad = False
                fast_weights[key] = fast_weights[key] - \
                    update_lr * grad_list[ind]

            resulta = net(inputa, labela, True, args, fast_weights)
            resulta[0].persistent = True
            resulta[1].persistent = True
            task_lossa_var.append(resulta[0])
            task_accuracya_var.append(resulta[1])

        # Loss on queries is calculated only at the end of the inner loop
        # Following the original implementation, 
        # we always use batch stats for batch normalization even in a test phase
        resultb = net(inputb, labelb, True, args, fast_weights)

        # Forward calculation
        result_all = F.sink(resulta[0], resulta[1], resultb[0], resultb[1])
        result_all.forward()

        if meta_training:
            # Backward calculation
            lossb = resultb[0] / data_generator.batch_size
            lossb.backward()  # gradients on weights are automatically accumlated

        task_lossa = []
        task_accuracya = []
        for j in range(num_updates + 1):
            task_accuracya_var[j].forward()
            task_lossa.append(task_lossa_var[j].d)
            task_accuracya.append(task_accuracya_var[j].d)

        lossesa.append(task_lossa)
        lossesb.append(resultb[0].d)
        accuraciesa.append(task_accuracya)
        accuraciesb.append(resultb[1].d)

    return lossesa, lossesb, accuraciesa, accuraciesb


def meta_train(exp_string, monitor, args):
    # Set monitors
    monitor_loss = MonitorSeries(
        'Training loss', monitor, interval=args.print_interval, verbose=False)
    monitor_valid_err = MonitorSeries(
        'Validation error', monitor, interval=args.test_print_interval, verbose=False)

    # Load data
    if args.datasource == 'omniglot':
        shape_x = (1, 28, 28)
        train_data, valid_data, _ = load_omniglot(
            os.path.join(args.dataset_root, 'omniglot/data/'), shape_x)
    else:
        raise ValueError('Unrecognized data source.')

    train_data_generator = DataGenerator(
        args.num_classes, args.train_num_shots, args.train_num_queries, shape_x, train_data, args.meta_batch_size)
    valid_data_generator = DataGenerator(
        args.num_classes, args.num_shots, args.num_queries, shape_x, valid_data, args.meta_batch_size)

    # Build training models
    # a: training data for inner gradient, b: test data for meta gradient
    inputa_t = nn.Variable((train_data_generator.num_classes *
                            train_data_generator.num_shots, ) + train_data_generator.shape_x)
    inputb_t = nn.Variable((train_data_generator.num_classes *
                            train_data_generator.num_queries, ) + train_data_generator.shape_x)
    labela_t = nn.Variable(
        (train_data_generator.num_classes * train_data_generator.num_shots, 1))
    labelb_t = nn.Variable(
        (train_data_generator.num_classes * train_data_generator.num_queries, 1))

    # Build evaluation models
    # a: training data for inner gradient, b: test data for meta gradient
    inputa_v = nn.Variable((valid_data_generator.num_classes *
                            valid_data_generator.num_shots, ) + valid_data_generator.shape_x)
    inputb_v = nn.Variable((valid_data_generator.num_classes *
                            valid_data_generator.num_queries, ) + valid_data_generator.shape_x)
    labela_v = nn.Variable(
        (valid_data_generator.num_classes * valid_data_generator.num_shots, 1))
    labelb_v = nn.Variable(
        (valid_data_generator.num_classes * valid_data_generator.num_queries, 1))

    with nn.parameter_scope('meta'):
        # Set weights
        _ = net(inputa_t, labela_t, True, args)  # only definition of weights
        weights = nn.get_parameters()

        # Setup solver
        solver = S.Adam(args.meta_lr)
        solver.set_parameters(weights)

    if args.num_updates > 1:
        print("[WARNING]: A number of updates in an inner loop is changed from " +
              str(args.updates) + " to 1")
        args.num_updates = 1

    print('Done initializing, starting training.')

    # Training loop
    for itr in range(1, args.metatrain_iterations + 1):
        solver.zero_grad()
        lossesa, lossesb, accuraciesa, accuraciesb = inner_train_test(
            inputa_t, inputb_t, labela_t, labelb_t, train_data_generator, True, args)
        solver.update()

        # Evaluation
        if itr % args.print_interval == 0:
            preaccuracies = np.mean(accuraciesa, axis=0)
            postaccuracy = np.mean(accuraciesb, axis=0)
            print_str = 'Iteration {}: '.format(itr)
            for j in range(len(preaccuracies)):
                print_str += ' %.4f ->' % preaccuracies[j]
            print_str += '->-> %.4f (final accuracy at queries)' % postaccuracy
            print(print_str)
            monitor_loss.add(itr, np.mean(lossesb, axis=0))

        if itr % args.test_print_interval == 0:
            # Inner training & testing
            lossesa, lossesb, accuraciesa, accuraciesb = inner_train_test(
                inputa_v, inputb_v, labela_v, labelb_v, valid_data_generator, False, args)

            # Validation
            preaccuracies = np.mean(accuraciesa, axis=0)
            postaccuracy = np.mean(accuraciesb, axis=0)
            print_str = 'Validation results: '
            for j in range(len(preaccuracies)):
                print_str += ' %.4f ->' % preaccuracies[j]
            print_str += '->-> %.4f (final accuracy at queries)' % postaccuracy
            print(print_str)
            monitor_valid_err.add(itr, (1.0 - postaccuracy) * 100.0)

        if itr % args.save_interval == 0:
            nn.save_parameters(os.path.join(
                args.logdir, exp_string, 'params{}.h5'.format(itr)))

    if itr % args.save_interval != 0:
        nn.save_parameters(os.path.join(
            args.logdir, exp_string, 'params{}.h5'.format(itr)))


def meta_test(exp_string, monitor, args):
    # Set monitors
    monitor_test_err = MonitorSeries('Test error', monitor, verbose=False)
    monitor_test_conf = MonitorSeries(
        'Test error confidence', monitor, verbose=False)

    # Load data
    if args.datasource == 'omniglot':
        shape_x = (1, 28, 28)
        _, _, test_data = load_omniglot(
            os.path.join(args.dataset_root, 'omniglot/data/'), shape_x)
    else:
        raise ValueError('Unrecognized data source.')

    test_data_generator = DataGenerator(
        args.num_classes, args.num_shots, args.num_queries, shape_x, test_data, 1)

    # Build testing models
    # a: training data for inner gradient, b: test data for meta gradient
    inputa_t = nn.Variable((test_data_generator.num_classes *
                            test_data_generator.num_shots, ) + test_data_generator.shape_x)
    inputb_t = nn.Variable((test_data_generator.num_classes *
                            test_data_generator.num_queries, ) + test_data_generator.shape_x)
    labela_t = nn.Variable(
        (test_data_generator.num_classes * test_data_generator.num_shots, 1))
    labelb_t = nn.Variable(
        (test_data_generator.num_classes * test_data_generator.num_queries, 1))

    # Restore model weights
    param_file = os.path.join(args.logdir, exp_string, args.param_file)
    print('Restoring model weights from {}'.format(param_file))
    nn.load_parameters(param_file)

    np.random.seed(1)
    random.seed(1)

    print('Done initializing, starting testing.')

    metaval_accuracies = []

    for _ in range(args.num_test_points):
        # Inner training & testing
        lossesa, lossesb, accuraciesa, accuraciesb = inner_train_test(
            inputa_t, inputb_t, labela_t, labelb_t, test_data_generator, False, args)

        metaval_accuracies.extend(np.concatenate((np.asarray(accuraciesa).reshape(
            (1, -1)), np.asarray(accuraciesb).reshape(1, -1)), axis=1))

    metaval_accuracies = np.asarray(metaval_accuracies)
    means = np.mean(metaval_accuracies, axis=0)
    stds = np.std(metaval_accuracies, axis=0)
    ci95 = 1.96 * stds / np.sqrt(args.num_test_points)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))
    monitor_test_err.add(0, (1.0 - means[-1]) * 100.0)
    monitor_test_conf.add(0, ci95[-1])

    out_filename = os.path.join(args.logdir, exp_string, 'test_shot{}_stepsize{}.csv'.format(
        args.num_shots, args.update_lr))
    out_pkl = os.path.join(args.logdir, exp_string, 'test_shot{}_stepsize{}.pkl'.format(
        args.num_shots, args.update_lr))
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update{}'.format(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def augmentation(data):
    # This function generates augmented class and data
    augmented_data = np.zeros((data.shape[0] * 4,) + data.shape[1:])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            augmented_data[i*4][j] = data[i][j]
            augmented_data[i*4+1][j] = np.rot90(data[i][j], 1)
            augmented_data[i*4+2][j] = np.rot90(data[i][j], 2)
            augmented_data[i*4+3][j] = np.rot90(data[i][j], 3)

    return augmented_data


def load_omniglot(dataset_root, shape_x, interp='lanczos'):
    x_train, _ = np.load(os.path.join(
        dataset_root, 'train.npy'), allow_pickle=True)
    x_valid, _ = np.load(os.path.join(
        dataset_root, 'val.npy'), allow_pickle=True)
    x = np.concatenate((x_train, x_valid))
    from PIL import Image
    x_resized = np.zeros((x.shape[0], x.shape[1], shape_x[1], shape_x[2]))
    for xi, ri in zip(x, x_resized):
        for xij, rij in zip(xi, ri):
            rij[:] = 1.0 - \
                np.array(Image.fromarray(xij).resize(
                    shape_x[1:], resample=2)) / 255.0
    rng = np.random.RandomState(706)
    data = rng.permutation(x_resized)
    data = augmentation(data)
    data = data.reshape((shape_x[0],) + data.shape).transpose(1, 2, 0, 3, 4)
    # Randomly selected 1200 characters are used for training + validation
    # Since the characters are augmented with rotations by multiples of 90 degrees,
    #  the total number of the selected characters should be 1200*4
    train_data = data[:1100*4]
    val_data = data[1100*4:1200*4]
    test_data = data[1200*4:]
    return train_data, val_data, test_data


def main():
    args = get_args()

    # # Set context
    from nnabla.ext_utils import get_extension_context
    logger.info('Running in {}'.format(args.context))
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    if args.num_queries == -1:
        args.num_queries = args.num_shots
    if args.train_num_shots == -1:
        args.train_num_shots = args.num_shots
    if args.train_num_queries == -1:
        args.train_num_queries = args.num_queries
    if args.train_update_lr == -1:
        args.train_update_lr = args.update_lr

    exp_string = 'way_{}.shot_{}.bs_{}.numstep{}.lr{}.first_order{}'.format(
        args.num_classes, args.train_num_shots, args.meta_batch_size, args.num_updates, args.train_update_lr, args.first_order)

    if args.num_filters != 64:
        exp_string += 'hidden{}'.format(args.num_filters)
    if args.max_pool:
        exp_string += 'maxpool'
    if args.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif args.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    # Monitor outputs
    monitor = Monitor(os.path.join(args.logdir, exp_string))

    if not args.test:
        meta_train(exp_string, monitor, args)
    else:
        meta_test(exp_string, monitor, args)


if __name__ == '__main__':
    main()
