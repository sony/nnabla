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
from collections import namedtuple
from vgg19 import PretrainedVgg19
from utils import *
from datetime import datetime
from discriminator_arch import discriminator
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

# ESRGAN Generator model


def get_esrgan_gen(conf, train_gt, train_lq, fake_h):
    """
    Create computation graph and variables for ESRGAN Generator.
    """
    var_ref = nn.Variable(
        (conf.train.batch_size, 3, conf.train.gt_size, conf.train.gt_size))
    # Feature Loss (L1 Loss)
    load_vgg19 = PretrainedVgg19()
    real_fea = load_vgg19(train_gt)
    # need_grad set to False, to avoid BP to vgg19 network
    real_fea.need_grad = False
    fake_fea = load_vgg19(fake_h)
    feature_loss = F.mean(F.absolute_error(fake_fea, real_fea))
    feature_loss.persistent = True

    # Gan Loss Generator
    with nn.parameter_scope("dis"):
        pred_g_fake = discriminator(fake_h)
        pred_d_real = discriminator(var_ref)
    pred_d_real.persistent = True
    pred_g_fake.persistent = True
    unlinked_pred_d_real = pred_d_real.get_unlinked_variable()
    gan_loss = RelativisticAverageGanLoss(GanLoss())
    gan_loss_gen_out = gan_loss(unlinked_pred_d_real, pred_g_fake)
    loss_gan_gen = gan_loss_gen_out.generator_loss
    loss_gan_gen.persistent = True
    Model_gen = namedtuple('Model_gen',
                           ['train_gt', 'train_lq', 'var_ref', 'feature_loss', 'loss_gan_gen', 'pred_d_real',
                            'pred_g_fake'])
    return Model_gen(train_gt, train_lq, var_ref, feature_loss, loss_gan_gen, pred_d_real, pred_g_fake)


# ESRGAN Discriminator model
def get_esrgan_dis(fake_h, pred_d_real):
    """
    Create computation graph and variables for ESRGAN Discriminator.
    """
    with nn.parameter_scope("dis"):
        unlinked_fake_h = fake_h.get_unlinked_variable()
        pred_d_fake = discriminator(unlinked_fake_h)
    gan_loss = RelativisticAverageGanLoss(GanLoss())
    gan_loss_dis_out = gan_loss(pred_d_real, pred_d_fake)
    l_d_total = gan_loss_dis_out.discriminator_loss
    l_d_total.persistent = True
    Model_dis = namedtuple('Model_dis',
                           ['l_d_total'])
    return Model_dis(l_d_total)

# ESRGAN Monitors


def get_esrgan_monitors():
    """
    Create monitors for displaying and storing ESRGAN losses.
    """
    monitor_path = './nnmonitor' + str(datetime.now().strftime("%Y%m%d%H%M%S"))
    monitor = Monitor(monitor_path)
    monitor_feature_g = MonitorSeries(
        'l_g_fea per iteration', monitor, interval=100)
    monitor_gan_g = MonitorSeries(
        'l_g_gan per iteration', monitor, interval=100)
    monitor_gan_d = MonitorSeries(
        'l_d_total per iteration', monitor, interval=100)
    monitor_d_real = MonitorSeries(
        'D_real per iteration', monitor, interval=100)
    monitor_d_fake = MonitorSeries(
        'D_fake per iteration', monitor, interval=100)
    Monitor_esrgan = namedtuple('Monitor_esrgan',
                                ['monitor_feature_g', 'monitor_gan_g', 'monitor_gan_d',
                                 'monitor_d_real', 'monitor_d_fake'])
    return Monitor_esrgan(monitor_feature_g, monitor_gan_g, monitor_gan_d, monitor_d_real, monitor_d_fake)
