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


import os
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from cycle_gan_data import cycle_gan_data_source, cycle_gan_data_iterator
import models
from helpers import MonitorImageWithName


def test(args):
    # Settings
    b, c, h, w = 1, 3, 256, 256
    beta1 = 0.5
    beta2 = 0.999
    lambda_recon = args.lambda_recon
    lambda_idt = args.lambda_idt
    base_lr = args.learning_rate
    init_method = args.init_method

    # Context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Inputs
    x_real_test = nn.Variable([b, c, h, w], need_grad=False)
    y_real_test = nn.Variable([b, c, h, w], need_grad=False)

    # Models for test
    nn.load_parameters(args.model_load_path)
    y_fake_test = models.g(
        x_real_test, unpool=args.unpool, init_method=init_method)
    x_fake_test = models.f(
        y_real_test, unpool=args.unpool, init_method=init_method)
    y_fake_test.persistent, x_fake_test.persistent = True, True
    # Reconstruct
    x_recon_test = models.f(
        y_fake_test, unpool=args.unpool, init_method=init_method)
    y_recon_test = models.g(
        x_fake_test, unpool=args.unpool, init_method=init_method)

    # Datasets
    rng = np.random.RandomState(313)
    ds_test_B = cycle_gan_data_source(
        args.dataset, train=False, domain="B", shuffle=False, rng=rng)
    ds_test_A = cycle_gan_data_source(
        args.dataset, train=False, domain="A", shuffle=False, rng=rng)
    di_test_B = cycle_gan_data_iterator(ds_test_B, args.batch_size)
    di_test_A = cycle_gan_data_iterator(ds_test_A, args.batch_size)

    # Monitors
    monitor = Monitor(args.monitor_path)

    def make_monitor_image(name):
        # return MonitorImageWithName(name, monitor, interval=1,
        #                         normalize_method=lambda x: (x + 1.0) * 127.5)
        return MonitorImageWithName(name, monitor, interval=1,
                                    normalize_method=lambda x: x + 1.0)
    monitor_test_gx = make_monitor_image('fake_images_test_A')
    monitor_test_fy = make_monitor_image('fake_images_test_B')
    monitor_test_x_recon = make_monitor_image('fake_images_recon_test_B')
    monitor_test_y_recon = make_monitor_image('fake_images_recon_test_A')

    # Validation for B
    logger.info("Validation for B")
    for i in range(di_test_A.size):
        name = ds_test_A.filename_list[i]
        logger.info("generating a fake of {}".format(name))
        y_data, _ = di_test_A.next()
        y_real_test.d = y_data
        y_recon_test.forward(clear_buffer=True)
        monitor_test_fy.add(name, x_fake_test.d)
        monitor_test_y_recon.add(name, y_recon_test.d)

    # Validation for A
    logger.info("Validation for A")
    for i in range(di_test_B.size):
        name = ds_test_B.filename_list[i]
        logger.info("generating a fake of {}".format(name))
        x_data, _ = di_test_B.next()
        x_real_test.d = x_data
        x_recon_test.forward(clear_buffer=True)
        monitor_test_gx.add(name, y_fake_test.d)
        monitor_test_x_recon.add(name, x_recon_test.d)


def main():
    args = get_args()
    save_args(args)
    test(args)


if __name__ == '__main__':
    main()
