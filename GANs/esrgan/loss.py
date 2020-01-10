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

import nnabla.functions as F
__all__ = [
    'GanLossContainer',
    'GanLoss',
    'RelativisticAverageGanLoss',
]


class GanLossContainer(object):
    '''
    A container class of GAN outputs from `GanLoss` classes.
    Attributes:
        loss_dr (~nnabla.Variable): Discriminator loss for real data.
        loss_df (~nnabla.Variable): Discriminator loss for fake data.
        loss_gr (~nnabla.Variable):
            Generator loss for real data. This is usually set as `None`
            because generator parameters are independent of real data in
            standard GAN losses. Exceptions for examples are relativistic
            GAN losses.
        loss_gf (~nnabla.Variable): Generator loss for fake data.
    Property:
        discriminator_loss (~nnabla.Variable):
            Returns `loss_dr + loss_df`. It is devided by 2 if loss_gr is
            not `None`.
        generator_loss (~nnabla.Variable): Returns `loss_gr + loss_gf`
    This class implements `+` (add) operators (radd as well) for the
    following operands.
    * `GanLossContainer + GanLossContainer`: Losses from containers are
      added up for each of loss_dr, loss_df, loss_gr, and loss_gf
      and a new GanLossContainer object is returned.
    * `GanLossContainer + None`: `None` is ignored and the given
      `GanLossContainer` is returned.
    '''

    def __init__(self, loss_dr, loss_df, loss_gr, loss_gf):
        self.loss_dr = loss_dr
        self.loss_df = loss_df
        self.loss_gr = loss_gr
        self.loss_gf = loss_gf
        self.d_div_factor = 2.0
        self._discriminator_loss = None
        self._generator_loss = None

    def set_persistent(self, b):
        self.loss_dr.persistent = b
        self.loss_df.persistent = b
        if self.loss_gr is not None:
            self.loss_gr.persistent = b
        self.loss_gf.persistent = b

    @property
    def generator_loss(self):
        if self._generator_loss is not None:
            return self._generator_loss
        self._generator_loss = self.loss_gf
        if self.loss_gr is not None:
            self._generator_loss += self.loss_gr
            self._generator_loss /= 2
        return self._generator_loss

    @property
    def discriminator_loss(self):
        if self._discriminator_loss is not None:
            return self._discriminator_loss
        self._discriminator_loss = \
            (self.loss_dr + self.loss_df) \
            / self.d_div_factor
        return self._discriminator_loss

    def __add__(self, rhs):
        if rhs is None:
            return self
        assert isinstance(rhs, GanLossContainer)
        loss_dr = self.loss_dr + rhs.loss_dr
        loss_df = self.loss_df + rhs.loss_df
        loss_gr = None
        if self.loss_gr is not None:
            loss_gr = self.loss_gr
        if rhs.loss_gr is not None:
            if loss_gr is None:
                loss_gr = rhs.loss_gr
            else:
                loss_gr += rhs.loss_gr
        loss_gf = self.loss_gf + rhs.loss_gf
        return GanLossContainer(loss_dr, loss_df, loss_gr, loss_gf)

    def __radd__(self, rhs):
        return self.__add__(rhs)


class BaseGanLoss(object):
    '''
    A base class of GAN loss functions.
    This class object offers a callable method which takes discriminator
    output variables from both real and fake, and returns a `GanLossContainer` which
    holds discreminator and generator loss values as computation graph
    variables.
    GAN loss functions for discriminator :math:`L_D` and generator :math:`L_G` can be written in the following generalized form
    .. math::
        L_D &= \mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[L^r_D \left(D(x_r)\right)\right] + \mathop{\mathbb{E}}_{x_f \sim G}\left[L^f_D \left( D(x_f) \right)\right] \\
        L_G &= \mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[L^r_G \left( D(x_r) \right)\right] + \mathop{\mathbb{E}}_{x_f \sim G} \left[L^f_G \left( D(x_f) \right)\right]
    where :math:`L^r_D` and :math:`L^r_G` are loss functions of real data :math:`x_r` sampled from dataset :math:`P_{\rm data}` for discriminator and generator respectively, and :math:`L^f_D` and :math:`L^f_G` are for fake data :math:`x_f` generated from the current generator :math:`G`. Those functions take discriminator outputs :math:`D(\cdot)` as inputs.
    In most of GAN variants (with some exceptions), those loss functions can be defined as the following symmetric form
    .. math::
        L^r_D(d) = l^+(d)
        L^f_D(d) = l^-(d)
        L^r_G(d) = l^-(d)
        L^f_G(d) = l^+(d)
    where :math:`l^+` is a loss function which encourages the discriminator
    output to be high while :math:`l^-` to be low.
    Different :math:`l^+` and :math:`l^-` give different types of GAN losses.
    For example, the Least Square GAN (LSGAN) is derived from
    .. math::
        l^+(d) &= (d - 1)^2 \\
        l^-(d) &= (d - 0)^2.
    Any derived class must implement both :math:`l^+(d)` :math:`l^-(d)` as `def _loss_plus(self, d)` and `def _loss_minus(self, d)` for :math:`l^+(d)` and :math:`l^-(d)` respectively, then the overall loss function is defined by the symmetric form explained above.
    Note:
        The loss term for real data of generator loss :math:`\mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[l^- \left( D(x_r) \right)\right]` is usually omitted at computation graph because generator model is not dependent of that term. If you want to obtain it as outputs, call a method `use_generator_loss_real(True)` to enable it.
    '''

    def __init__(self):

        self._use_generator_loss_for_real = False

    def use_generator_loss_for_real(self, use):
        '''
        Whether or not to compute generator loss for real data. This is
        originally set as False, because the generator model which
        we would like to train is not dependent of real data, so we don't
        have to consider the generator loss for real data as training loss.
        Args:
            use (bool):
                If True, the generator loss for real data is taken into
                account.
        '''
        self._use_generator_loss_for_real = use

    def _mean(self, loss):
        '''
        Reduction function to obtain a scalar value for each GAN loss.
        '''
        return F.mean(loss)

    def _loss_plus(self, dout):
        raise NotImplementedError('Not implemented.')

    def _loss_minus(self, dout):
        raise NotImplementedError('Not implemented.')

    def _loss_dis_real(self, dout):
        return self._loss_plus(dout)

    def _loss_dis_fake(self, dout):
        return self._loss_minus(dout)

    def _loss_gen_real(self, dout):
        return self._loss_minus(dout)

    def _loss_gen_fake(self, dout):
        return self._loss_plus(dout)

    def __call__(self, d_r, d_f):
        '''
        Get GAN losses given disriminator outputs of both real and fake.
        Args:
            d_r (~nnabla.Variable): Discriminator output of real data, `D(x_real)`.
            d_f (~nnabla.Variable): Discriminator output of fake data, `D(x_fake)`.
        Note:
            The discriminator scores which are fed into this must be
            pre-activation values, that is `[-inf, inf]`.
        Returns: GanLossContainer
        '''
        # L_D
        loss_dr = self._mean(self._loss_dis_real(d_r))
        loss_df = self._mean(self._loss_dis_fake(d_f))

        # L_G
        loss_gr = None
        if self._use_generator_loss_for_real:
            loss_gr = self._mean(self._loss_gen_real(d_r))
        loss_gf = self._mean(self._loss_gen_fake(d_f))
        return GanLossContainer(loss_dr, loss_df, loss_gr, loss_gf)


class GanLoss(BaseGanLoss):
    '''
    Standard GAN loss defined as

    .. math::
        l^+(d) &= \ln \sigma (d) \\
        l^-(d) &= \ln \left(1 - \sigma (d )\right)

    in a generalized form described in `BaseGanLoss` documentation. Here, :math:`\sigma` is Sigmoid function :math:`\sigma(d) = \frac{1}{1 + e^{-d}}` to interpret input as probability.

    References:

        `Ian J. Goodfellow et. al.
        Generative Adversarial Networks
        <https://arxiv.org/abs/1406.2661>`_

    '''

    def _loss_plus(self, dout):
        return -F.log_sigmoid(dout)

    def _loss_minus(self, dout):
        return -F.log_sigmoid(-dout)


class RelativisticAverageGanLoss(object):
    '''
    Relativistic Average GAN (RaGAN) Loss.

    Args:
        gan_loss (BaseGanLoss): A GAN loss.
        average (bool): If False, averaging is omitted. Hence it becomes Relativistic GAN.

    References:
        `Alexia Jolicoeur-Martineau.
        Relativistic Average GAN.
        <https://arxiv.org/pdf/1807.00734.pdf>`_

    '''

    def __init__(self, gan_loss, average=True):
        import copy
        assert isinstance(gan_loss, BaseGanLoss)
        gan_loss = copy.copy(gan_loss)
        # Relativistic GANs require generator loss for real
        gan_loss.use_generator_loss_for_real(True)
        self._gan_loss = gan_loss
        self._average = average

    def _average_func(self, d_values):
        if not self._average:
            return d_values
        return F.mean(d_values, keepdims=True)

    def __call__(self, d_r, d_f):

        rel_d_r = d_r - self._average_func(d_f)
        rel_d_f = d_f - self._average_func(d_r)
        return self._gan_loss(rel_d_r, rel_d_f)
