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
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter_or_create
from nnabla.ext_utils import get_extension_context
from nnabla.parametric_functions import parametric_function_api

from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from functools import reduce

def spectral_normalization_for_conv(w, itr=1, eps=1e-12, test=False):
    w_shape = w.shape
    W_sn = get_parameter_or_create("W_sn", w_shape, ConstantInitializer(0), False)
    if test:
        return W_sn

    d0 = w.shape[0]            # Out
    d1 = np.prod(w.shape[1:])  # In
    w = F.reshape(w, [d0, d1], inplace=False)
    u0 = get_parameter_or_create("singular-vector", [d0], NormalInitializer(), False)
    u = F.reshape(u0, [1, d0])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(u, w)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [d1, 1])
        # u
        u = F.affine(w, v)
        u = F.div2(u, F.pow_scalar(F.sum(F.pow_scalar(u, 2.), keepdims=True) + eps, 0.5))
        u = F.reshape(u, [1, d0])
    # Iterate
    u = F.identity(u, outputs=[u0.data])
    u.persistent = True
    # No grad
    u.need_grad = False
    v.need_grad = False
    # Spectral normalization
    wv = F.affine(w, v)
    sigma = F.affine(u, wv)
    w_sn = F.div2(w, sigma)
    w_sn = F.reshape(w_sn, w_shape)
    w_sn = F.identity(w_sn, outputs=[W_sn.data])
    w_sn.persistent = True
    return w_sn


def spectral_normalization_for_affine(w, itr=1, eps=1e-12, input_axis=1, test=False):
    W_sn = get_parameter_or_create("W_sn", w.shape, ConstantInitializer(0), False)
    if test:
        return W_sn

    d0 = np.prod(w.shape[0:-1])  # In
    d1 = np.prod(w.shape[-1])   # Out
    u0 = get_parameter_or_create("singular-vector", [d1], NormalInitializer(), False)
    u = F.reshape(u0, [d1, 1])
    # Power method
    for _ in range(itr):
        # v
        v = F.affine(w, u)
        v = F.div2(v, F.pow_scalar(F.sum(F.pow_scalar(v, 2.), keepdims=True) + eps, 0.5))
        v = F.reshape(v, [1, d0])
        # u
        u = F.affine(v, w)
        u = F.div2(u, F.pow_scalar(F.sum(F.pow_scalar(u, 2.), keepdims=True) + eps, 0.5))
        u = F.reshape(u, [d1, 1])
    # Iterate
    u = F.identity(u, outputs=[u0.data])
    u.persistent = True
    # No grad
    u.need_grad = False
    v.need_grad = False
    # Spectral normalization
    wv = F.affine(v, w)
    sigma = F.affine(wv, u)
    sigma = F.broadcast(F.reshape(sigma, [1 for _ in range(len(w.shape))]), w.shape)
    w_sn = F.div2(w, sigma, outputs=[W_sn.data])
    w_sn.persistent = True
    return w_sn


@parametric_function_api("sn_conv")
def convolution(inp, outmaps, kernel,
                pad=None, stride=None, dilation=None, group=1,
                itr=1, 
                w_init=None, b_init=None,
                base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                sn=True, test=False, init_scale=1.0):
                
    """
    """
    if w_init is None:
        l, u = calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel))
        l, u = init_scale * l, init_scale * u
        w_init = UniformInitializer((l, u), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)
    w_sn = spectral_normalization_for_conv(w, itr=itr, test=test) if sn else w
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.convolution(inp, w_sn, b, base_axis, pad, stride, dilation, group)
    

@parametric_function_api("sn_affine")
def affine(inp, n_outmaps,
           base_axis=1,
           w_init=None, b_init=None,
           itr=1, 
           fix_parameters=False, rng=None, with_bias=True,
           sn=True, test=False):
    """
    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    w_sn = spectral_normalization_for_affine(w, itr=itr, test=test) if sn else w
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.affine(inp, w_sn, b, base_axis)
    

@parametric_function_api("embed")
def embed(inp, n_inputs, n_features, initializer=None,
          itr=1, fix_parameters=False, sn=True, test=False):
    """
    """
    w = get_parameter_or_create("W", [n_inputs, n_features],
                                initializer, not fix_parameters)
    w_sn = spectral_normalization_for_affine(w, itr=itr, test=test) if sn else w
    return F.embed(inp, w_sn)
    

def BN(h, decay_rate=0.999, test=False):
    """Batch Normalization"""
    return PF.batch_normalization(h, decay_rate=decay_rate, batch_stat=not test)


@parametric_function_api("ccbn")
def CCBN(h, y, n_classes, decay_rate=0.999, test=False, fix_parameters=False, coefs=[1.0]):
    """Categorical Conditional Batch Normaliazation"""
    # Call the batch normalization once
    shape_stat = [1 for _ in h.shape]
    shape_stat[1] = h.shape[1]
    gamma_tmp = nn.Variable.from_numpy_array(np.ones(shape_stat))
    beta_tmp = nn.Variable.from_numpy_array(np.zeros(shape_stat))
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0.0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(1.0), False)
    h = F.batch_normalization(h, beta_tmp, gamma_tmp, mean, var,
                              decay_rate=decay_rate, batch_stat=not test)

    # Condition the gamma and beta with the class label
    b, c = h.shape[0:2]
    def embed_func(y, initializer):
        if type(y) != list:
            o = embed(y, n_classes, c, initializer=initializer, sn=False, test=test)
        else:
            y_list = y
            o = reduce(lambda x, y: x + y, 
                       [coef * embed(
                           y, n_classes, c, initializer=initializer, sn=False, test=test) \
                        for coef, y in zip(coefs, y_list)])
        return o
    with nn.parameter_scope("gamma"):
        gamma = embed_func(y, ConstantInitializer(1.0))
        gamma = F.reshape(gamma, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        gamma = F.broadcast(gamma, h.shape)
    with nn.parameter_scope("beta"):
        beta = embed_func(y, ConstantInitializer(0.0))
        beta = F.reshape(beta, [b, c] + [1 for _ in range(len(h.shape[2:]))])
        beta = F.broadcast(beta, h.shape)
    return gamma * h + beta


@parametric_function_api("attn")
def attnblock(h, r=8, fix_parameters=False, sn=True, test=False):
    """Attention block"""
    x = h

    # 1x1 convolutions
    b, c, s0, s1 = h.shape
    c_r = c // r
    assert c_r > 0
    f_x = convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="f",
                      with_bias=False, sn=sn, test=test)
    g_x = convolution(h, c_r, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="g",
                      with_bias=False, sn=sn, test=test)
    h_x = convolution(h, c, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="h",
                      with_bias=False, sn=sn, test=test)

    # Attend 
    attn = F.batch_matmul(f_x.reshape([b, c_r, -1]), g_x.reshape([b, c_r, -1]), transpose_a=True)
    attn = F.softmax(attn, 1)
    h_x = h_x.reshape([b, c, -1])
    o = F.batch_matmul(h_x, attn)
    o = F.reshape(o, [b, c, s0, s1])

    # Shortcut
    gamma = get_parameter_or_create("gamma", [1, 1, 1, 1], ConstantInitializer(0.), not fix_parameters)
    y = gamma * o + x
    return y


def resblock_g(h, y, scopename, 
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               upsample=True, test=False, sn=True, coefs=[1.0]):
    """Residual block for generator"""

    s = h
    _, c, _, _ = h.shape
    with nn.parameter_scope(scopename):
        # BN -> Relu -> Upsample -> Conv
        with nn.parameter_scope("conv1"):
            h = CCBN(h, y, n_classes, test=test, coefs=coefs)
            h = F.relu(h, inplace=True)
            if upsample:
                h = F.unpooling(h, kernel=(2, 2))
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
        
        # BN -> Relu -> Conv
        with nn.parameter_scope("conv2"):
            h = CCBN(h, y, n_classes, test=test, coefs=coefs)
            h = F.relu(h, inplace=True)
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
            
        # Shortcut: Upsample -> Conv
        if upsample:
            s = F.unpooling(s, kernel=(2, 2))
        if c != maps or upsample:
            with nn.parameter_scope("shortcut"):
                s = convolution(s, maps, kernel=(1, 1), pad=(0, 0), stride=(1, 1), 
                                with_bias=True, sn=sn, test=test)
    return F.add2(h, s, True)


def resblock_d(h, y, scopename,
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               downsample=True, test=False, sn=True):
    """Residual block for discriminator"""
    s = h
    _, c, _, _ = h.shape
    assert maps // 2 == c or maps == c
    maps1 = c if maps // 2 == c else maps
    maps2 = maps
    with nn.parameter_scope(scopename):
        # LeakyRelu -> Conv
        with nn.parameter_scope("conv1"):
            #h = F.leaky_relu(h, 0.2, False)
            h = F.relu(h, False)
            h = convolution(h, maps1, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
        
        # LeakyRelu -> Conv -> Downsample
        with nn.parameter_scope("conv2"):
            #h = F.leaky_relu(h, 0.2, True)
            h = F.relu(h, True)
            h = convolution(h, maps2, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
            if downsample:
                h = F.average_pooling(h, kernel=(2, 2))
            
        # Shortcut: Conv -> Downsample
        if c != maps2 or downsample:
            with nn.parameter_scope("shortcut"):
                s = convolution(s, maps2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), 
                                with_bias=True, sn=sn, test=test)
        if downsample:
            s = F.average_pooling(s, kernel=(2, 2))
    return F.add2(h, s, True)
    #return F.add2(h, s)


def optblock_d(h, y, scopename,
               n_classes, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), 
               downsample=True, test=False, sn=True):
    """Optimized block for discriminator"""
    s = h
    _, c, _, _ = h.shape
    with nn.parameter_scope(scopename):
        # Conv
        with nn.parameter_scope("conv1"):
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
        
        # ReLU -> Conv
        with nn.parameter_scope("conv2"):
            #h = F.leaky_relu(h, 0.2, True)
            h = F.relu(h, True)
            h = convolution(h, maps, kernel=kernel, pad=pad, stride=stride, 
                            with_bias=True, sn=sn, test=test, init_scale=np.sqrt(2))
            if downsample:
                h = F.average_pooling(h, kernel=(2, 2))
            
        # Shortcut: Conv -> Downsample
        with nn.parameter_scope("shortcut"):
            if downsample:
                s = F.average_pooling(s, kernel=(2, 2))
            s = convolution(s, maps, kernel=(1, 1), pad=(0, 0), stride=(1, 1), 
                            with_bias=True, sn=sn, test=test)
    return F.add2(h, s, True)


def generator(z, y, scopename="generator", 
              maps=1024, n_classes=1000, s=4, test=False, sn=True, coefs=[1.0]):
    with nn.parameter_scope(scopename):
        # Affine
        h = affine(z, maps * s * s, with_bias=True, sn=sn, test=test)
        h = F.reshape(h, [h.shape[0]] + [maps, s, s])
        # Resblocks
        h = resblock_g(h, y, "block-1", n_classes, maps // 1, test=test, sn=sn, coefs=coefs)
        h = resblock_g(h, y, "block-2", n_classes, maps // 2, test=test, sn=sn, coefs=coefs)
        h = resblock_g(h, y, "block-3", n_classes, maps // 4, test=test, sn=sn, coefs=coefs)
        h = attnblock(h, sn=sn, test=test)
        h = resblock_g(h, y, "block-4", n_classes, maps // 8, test=test, sn=sn, coefs=coefs)
        h = resblock_g(h, y, "block-5", n_classes, maps // 16, test=test, sn=sn, coefs=coefs)
        # Last convoltion
        h = BN(h, test=test)
        h = F.relu(h)
        h = convolution(h, 3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), sn=sn, test=test)
        x = F.tanh(h)
    return x


def discriminator(x, y, scopename="discriminator", 
                  maps=64, n_classes=1000, s=4, test=False, sn=True):
    with nn.parameter_scope(scopename):
        # Resblocks
        h = optblock_d(x, y, "block-1", n_classes, maps * 1, test=test, sn=sn)
        h = resblock_d(h, y, "block-2", n_classes, maps * 2, test=test, sn=sn)
        h = attnblock(h, sn=sn, test=test)
        h = resblock_d(h, y, "block-3", n_classes, maps * 4, test=test, sn=sn)
        h = resblock_d(h, y, "block-4", n_classes, maps * 8, test=test, sn=sn)
        h = resblock_d(h, y, "block-5", n_classes, maps * 16, test=test, sn=sn)
        h = resblock_d(h, y, "block-6", n_classes, maps * 16, downsample=False, test=test, sn=sn)
        # Last affine
        #h = F.leaky_relu(h, 0.2, True)
        h = F.relu(h, True)
        h = F.sum(h, axis=(2, 3))
        o0 = affine(h, 1, sn=sn, test=test)
        # Project discriminator
        l, u = calc_uniform_lim_glorot(n_classes, maps * 16)
        e = embed(y, n_classes, maps * 16,
                  initializer=UniformInitializer((l, u)), name="projection",
                  sn=sn, test=test)
        o1 = F.sum(h * e, axis=1, keepdims=True)
    return o0 + o1


def gan_loss(p_fake, p_real=None):
    """Hinge loss"""
    if p_real is None:
        return -F.mean(p_fake)
    #return F.maximum_scalar(1.0 - p_real, 0.0) + F.maximum_scalar(1.0 + p_fake, 0.0)
    return F.mean(F.relu(1.0 - p_real)) + F.mean(F.relu(1.0 + p_fake))

    
if __name__ == '__main__':
    b, c, h, w = 4, 3, 128, 128
    latent = 128

    
