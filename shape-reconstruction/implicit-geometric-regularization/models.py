# Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

from __future__ import division

import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.initializer as I
import nnabla.functions as F
import nnabla.parametric_functions as PF

from functools import partial


class MLP():

    def __init__(self, dims=512, ldims=0, test=False, act="leaky_relu"):
        self.dims = dims
        self.ldims = ldims
        self.test = test
        self.act_map = dict(
            softplus=partial(self.softplus, beta=100),
            relu=F.relu,  # got nan
            leaky_relu=F.leaky_relu)
        self.act = self.act_map[act]

    def softplus(self, x, beta=1):
        return (1.0 / beta) * F.log(1.0 + F.exp(beta * x))

    def affine_act(self, x, dims, name):
        c = x.shape[1]
        s = I.calc_normal_std_he_forward(c, dims)
        w_init = I.NormalInitializer(s, )
        return self.act(PF.affine(x, dims, w_init=w_init, name=name))

    def last_affine(self, x, dims, name):
        c = x.shape[1]
        l, u = I.calc_uniform_lim_glorot(c, 1)
        w_init = I.UniformInitializer((l, u))
        return PF.affine(x, 1, w_init=w_init, name=name)

    def __call__(self, x, z=None):
        b, c = x.shape[0:2]
        h = x
        h = self.affine_act(h, self.dims, name="fc0")
        h = self.affine_act(h, self.dims, name="fc1")
        h = self.affine_act(h, self.dims, name="fc2")
        h = self.affine_act(h, self.dims - (self.ldims + c), name="fc3")
        h = F.concatenate(
            *[x, z, h], axis=1) if z is not None else F.concatenate(*[x, h], axis=1)
        h = self.affine_act(h, self.dims, name="fc4")
        h = self.affine_act(h, self.dims, name="fc5")
        h = self.affine_act(h, self.dims, name="fc6")
        y = self.last_affine(h, 1, name="fc7")
        return y


if __name__ == '__main__':
    x = nn.Variable([1, 3])
    model = MLP()
    y = model(x)

    class PrintFunc():
        def __call__(sef, func):
            print(func.info.type_name)
            print("\tinp: {}".format([i.shape for i in func.inputs]))
            print("\tout: {}".format([o.shape for o in func.outputs]))
    y.visit(PrintFunc())
