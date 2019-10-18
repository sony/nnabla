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


import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
import functools
import random
import datetime
import json
import model
import loss

from nnabla.ext_utils import get_extension_context
from args import get_args
from dataloader import stargan_load_func, get_data_dict
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.utils.image_utils import imsave
from nnabla.utils.data_iterator import data_iterator_simple


def saveimage(path, img):
    img = (img * 0.5) + 0.5  # Normalize.
    imsave(path, img, channel_first=True)


def save_results(i, args, img_src, img_trg, lbl_src, lbl_trg, img_rec=None, is_training=True):
    if is_training:
        filenamebase = "Train_at_iter"
    else:
        filenamebase = "Test_at_iter"
    chosen_idx = np.random.randint(0, args.batch_size)
    target_attr_flags = lbl_trg.d[chosen_idx].reshape(
        lbl_trg.d[chosen_idx].size)
    target_domain = "_".join([attr for idx, attr in zip(
        target_attr_flags, args.selected_attrs) if bool(idx) is True])
    source_attr_flags = lbl_src.d[chosen_idx].reshape(
        lbl_src.d[chosen_idx].size)
    source_domain = "_".join([attr for idx, attr in zip(
        source_attr_flags, args.selected_attrs) if bool(idx) is True])
    source_x = img_src.d[chosen_idx]
    result_x = img_trg.d[chosen_idx]
    saveimage("{}/{}_{}_result_{}.png".format(args.monitor_path,
                                              filenamebase, i, target_domain), result_x)
    saveimage("{}/{}_{}_source_{}.png".format(args.monitor_path,
                                              filenamebase, i, source_domain), source_x)

    if img_rec:
        recon_x = img_rec.d[chosen_idx]
        saveimage("{}/{}_{}_recon_{}.png".format(args.monitor_path,
                                                 filenamebase, i, source_domain), recon_x)
    return


def train(args):
    if args.c_dim != len(args.selected_attrs):
        print("c_dim must be the same as the num of selected attributes. Modified c_dim.")
        args.c_dim = len(args.selected_attrs)

    # Dump the config information.
    config = dict()
    print("Used config:")
    for k in args.__dir__():
        if not k.startswith("_"):
            config[k] = getattr(args, k)
            print("'{}' : {}".format(k, getattr(args, k)))

    # Prepare Generator and Discriminator based on user config.
    generator = functools.partial(
        model.generator, conv_dim=args.g_conv_dim, c_dim=args.c_dim, repeat_num=args.g_repeat_num)
    discriminator = functools.partial(model.discriminator, image_size=args.image_size,
                                      conv_dim=args.d_conv_dim, c_dim=args.c_dim, repeat_num=args.d_repeat_num)

    x_real = nn.Variable(
        [args.batch_size, 3, args.image_size, args.image_size])
    label_org = nn.Variable([args.batch_size, args.c_dim, 1, 1])
    label_trg = nn.Variable([args.batch_size, args.c_dim, 1, 1])

    with nn.parameter_scope("dis"):
        dis_real_img, dis_real_cls = discriminator(x_real)

    with nn.parameter_scope("gen"):
        x_fake = generator(x_real, label_trg)
    x_fake.persistent = True  # to retain its value during computation.

    # get an unlinked_variable of x_fake
    x_fake_unlinked = x_fake.get_unlinked_variable()

    with nn.parameter_scope("dis"):
        dis_fake_img, dis_fake_cls = discriminator(x_fake_unlinked)

    # ---------------- Define Loss for Discriminator -----------------
    d_loss_real = (-1) * loss.gan_loss(dis_real_img)
    d_loss_fake = loss.gan_loss(dis_fake_img)
    d_loss_cls = loss.classification_loss(dis_real_cls, label_org)
    d_loss_cls.persistent = True

    # Gradient Penalty.
    alpha = F.rand(shape=(args.batch_size, 1, 1, 1))
    x_hat = F.mul2(alpha, x_real) + \
        F.mul2(F.r_sub_scalar(alpha, 1), x_fake_unlinked)

    with nn.parameter_scope("dis"):
        dis_for_gp, _ = discriminator(x_hat)
    grads = nn.grad([dis_for_gp], [x_hat])

    l2norm = F.sum(grads[0] ** 2.0, axis=(1, 2, 3)) ** 0.5
    d_loss_gp = F.mean((l2norm - 1.0) ** 2.0)

    # total discriminator loss.
    d_loss = d_loss_real + d_loss_fake + args.lambda_cls * \
        d_loss_cls + args.lambda_gp * d_loss_gp

    # ---------------- Define Loss for Generator -----------------
    g_loss_fake = (-1) * loss.gan_loss(dis_fake_img)
    g_loss_cls = loss.classification_loss(dis_fake_cls, label_trg)
    g_loss_cls.persistent = True

    # Reconstruct Images.
    with nn.parameter_scope("gen"):
        x_recon = generator(x_fake_unlinked, label_org)
    x_recon.persistent = True

    g_loss_rec = loss.recon_loss(x_real, x_recon)
    g_loss_rec.persistent = True

    # total generator loss.
    g_loss = g_loss_fake + args.lambda_rec * \
        g_loss_rec + args.lambda_cls * g_loss_cls

    # -------------------- Solver Setup ---------------------
    d_lr = args.d_lr  # initial learning rate for Discriminator
    g_lr = args.g_lr  # initial learning rate for Generator
    solver_dis = S.Adam(alpha=args.d_lr, beta1=args.beta1, beta2=args.beta2)
    solver_gen = S.Adam(alpha=args.g_lr, beta1=args.beta1, beta2=args.beta2)

    # register parameters to each solver.
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())

    # -------------------- Create Monitors --------------------
    monitor = Monitor(args.monitor_path)
    monitor_d_cls_loss = MonitorSeries(
        'real_classification_loss', monitor, args.log_step)
    monitor_g_cls_loss = MonitorSeries(
        'fake_classification_loss', monitor, args.log_step)
    monitor_loss_dis = MonitorSeries(
        'discriminator_loss', monitor, args.log_step)
    monitor_recon_loss = MonitorSeries(
        'reconstruction_loss', monitor, args.log_step)
    monitor_loss_gen = MonitorSeries('generator_loss', monitor, args.log_step)
    monitor_time = MonitorTimeElapsed("Training_time", monitor, args.log_step)

    # -------------------- Prepare / Split Dataset --------------------
    using_attr = args.selected_attrs
    dataset, attr2idx, idx2attr = get_data_dict(args.attr_path, using_attr)
    random.seed(313)  # use fixed seed.
    random.shuffle(dataset)  # shuffle dataset.
    test_dataset = dataset[-2000:]  # extract 2000 images for test

    if args.num_data:
        # Use training data partially.
        training_dataset = dataset[:min(args.num_data, len(dataset) - 2000)]
    else:
        training_dataset = dataset[:-2000]
    print("Use {} images for training.".format(len(training_dataset)))

    # create data iterators.
    load_func = functools.partial(stargan_load_func, dataset=training_dataset,
                                  image_dir=args.celeba_image_dir, image_size=args.image_size, crop_size=args.celeba_crop_size)
    data_iterator = data_iterator_simple(load_func, len(
        training_dataset), args.batch_size, with_file_cache=False, with_memory_cache=False)

    load_func_test = functools.partial(stargan_load_func, dataset=test_dataset,
                                       image_dir=args.celeba_image_dir, image_size=args.image_size, crop_size=args.celeba_crop_size)
    test_data_iterator = data_iterator_simple(load_func_test, len(
        test_dataset), args.batch_size, with_file_cache=False, with_memory_cache=False)

    # Keep fixed test images for intermediate translation visualization.
    test_real_ndarray, test_label_ndarray = test_data_iterator.next()
    test_label_ndarray = test_label_ndarray.reshape(
        test_label_ndarray.shape + (1, 1))

    # -------------------- Training Loop --------------------
    one_epoch = data_iterator.size // args.batch_size
    num_max_iter = args.max_epoch * one_epoch

    for i in range(num_max_iter):
        # Get real images and labels.
        real_ndarray, label_ndarray = data_iterator.next()
        label_ndarray = label_ndarray.reshape(label_ndarray.shape + (1, 1))
        x_real.d, label_org.d = real_ndarray, label_ndarray

        # Generate target domain labels randomly.
        rand_idx = np.random.permutation(label_org.shape[0])
        label_trg.d = label_ndarray[rand_idx]

        # ---------------- Train Discriminator -----------------
        # generate fake image.
        x_fake.forward(clear_no_need_grad=True)
        d_loss.forward(clear_no_need_grad=True)
        solver_dis.zero_grad()
        d_loss.backward(clear_buffer=True)
        solver_dis.update()

        monitor_loss_dis.add(i, d_loss.d.item())
        monitor_d_cls_loss.add(i, d_loss_cls.d.item())
        monitor_time.add(i)

        # -------------- Train Generator --------------
        if (i + 1) % args.n_critic == 0:
            g_loss.forward(clear_no_need_grad=True)
            solver_dis.zero_grad()
            solver_gen.zero_grad()
            x_fake_unlinked.grad.zero()
            g_loss.backward(clear_buffer=True)
            x_fake.backward(grad=None)
            solver_gen.update()
            monitor_loss_gen.add(i, g_loss.d.item())
            monitor_g_cls_loss.add(i, g_loss_cls.d.item())
            monitor_recon_loss.add(i, g_loss_rec.d.item())
            monitor_time.add(i)

            if (i + 1) % args.sample_step == 0:
                # save image.
                save_results(i, args, x_real, x_fake,
                             label_org, label_trg, x_recon)
                if args.test_during_training:
                    # translate images from test dataset.
                    x_real.d, label_org.d = test_real_ndarray, test_label_ndarray
                    label_trg.d = test_label_ndarray[rand_idx]
                    x_fake.forward(clear_no_need_grad=True)
                    save_results(i, args, x_real, x_fake, label_org,
                                 label_trg, None, is_training=False)

        # Learning rates get decayed
        if (i + 1) > int(0.5 * num_max_iter) and (i + 1) % args.lr_update_step == 0:
            g_lr = max(0, g_lr - (args.lr_update_step *
                                  args.g_lr / float(0.5 * num_max_iter)))
            d_lr = max(0, d_lr - (args.lr_update_step *
                                  args.d_lr / float(0.5 * num_max_iter)))
            solver_gen.set_learning_rate(g_lr)
            solver_dis.set_learning_rate(d_lr)
            print('learning rates decayed, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    # Save parameters and training config.
    param_name = 'trained_params_{}.h5'.format(
        datetime.datetime.today().strftime("%m%d%H%M"))
    param_path = os.path.join(args.model_save_path, param_name)
    nn.save_parameters(param_path)
    config["pretrained_params"] = param_name

    with open(os.path.join(args.model_save_path, "training_conf_{}.json".format(datetime.datetime.today().strftime("%m%d%H%M"))), "w") as f:
        json.dump(config, f)

    # -------------------- Translation on test dataset --------------------
    for i in range(args.num_test):
        real_ndarray, label_ndarray = test_data_iterator.next()
        label_ndarray = label_ndarray.reshape(label_ndarray.shape + (1, 1))
        x_real.d, label_org.d = real_ndarray, label_ndarray

        rand_idx = np.random.permutation(label_org.shape[0])
        label_trg.d = label_ndarray[rand_idx]

        x_fake.forward(clear_no_need_grad=True)
        save_results(i, args, x_real, x_fake, label_org,
                     label_trg, None, is_training=False)


def main():
    args = get_args()
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    train(args)


if __name__ == '__main__':
    main()
