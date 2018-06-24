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

from six import iteritems

import nnabla as nn
import numpy as np
from collections import OrderedDict


class RefSolver(object):

    def set_parameters(self, params):
        if not hasattr(self, 'params'):
            self.params = OrderedDict()
        for key, param in iteritems(params):
            param = param.d.copy()
            if key in self.params:
                continue
            self.params[key] = param
            self._set_state_impl(key, param)

    def _set_state_impl(self, key, param):
        pass

    def update(self, grads):
        for key, grad in iteritems(grads):
            param = self.params[key]
            self._update_impl(key, param, grad)

    def weight_decay(self, grads, decay_rate):
        for key, grad in iteritems(grads):
            param = self.params[key]
            grad[...] = grad + decay_rate * param


def solver_tester(rng, solver, ref_solver, solver_args=[], solver_kwargs={},
                  num_itr=5, decay=1e-4, atol=1e-6,
                  ctx=None, solver_name=None):
    if ctx is None:
        ctx = nn.Context()

    # Create params
    p1 = nn.Variable([2, 3, 4])
    p2 = nn.Variable([3, 4, 1, 2])
    p3 = nn.Variable([])

    params = OrderedDict([('zZzZ', p1), ('bbb', p2), ('asdfadfdasd', p3)])
    for p in params.values():
        p.d = rng.randn(*p.shape)
        p.g = rng.randn(*p.shape)

    with nn.context_scope(ctx):
        s = solver(*solver_args, **solver_kwargs)
    s.set_parameters(params)
    if solver_name is not None:
        assert s.name == solver_name

    ref_s = ref_solver(*solver_args, **solver_kwargs)
    ref_s.set_parameters(params)

    # Check weight decay.
    grad_copy = OrderedDict([(k, p.g.copy())
                             for k, p in iteritems(params)])
    s.weight_decay(decay)
    ref_s.weight_decay(grad_copy, decay)
    for p, ref_p in zip(params.values(), grad_copy.values()):
        assert np.allclose(ref_p, p.g, atol=atol)

    # Check solver udpate.
    for i in range(num_itr):
        grads = OrderedDict([(k, rng.randn(*p.shape))
                             for k, p in iteritems(params)])
        for k, g in iteritems(grads):
            params[k].g = g
        s.update()
        ref_s.update(grads)
        for p, ref_p in zip(params.values(), ref_s.params.values()):
            assert np.allclose(ref_p, p.d, atol=atol)

    # Check inf, nan, and inf/nan
    for v, method in zip([[np.inf], [np.nan], [np.inf, np.nan]],
                         [lambda s: s.check_inf_grad(),
                          lambda s: s.check_nan_grad(),
                          lambda s: s.check_inf_or_nan_grad()]):
        def set_value(p):
            p.g[...] = rng.choice(v + [-1, 0, 1],
                                  size=int(np.prod(p.shape)),
                                  replace=True).reshape(p.shape)
            if v[0] not in p.g:
                p.g.flat[rng.choice(np.arange(int(np.prod(p.shape))))] = v[0]
        for p in params.values():
            assert method(s) == False
            g = p.g.copy()
            set_value(p)
            assert method(s) == True
            p.g[...] = g

    # Rescale grad
    scale = 10.
    ref_grad = [p.g.copy() for p in params.values()]
    for p in params.values():
        p.g *= scale
    s.scale_grad(1. / scale)
    for ref, p in zip(ref_grad, params.values()):
        assert np.allclose(ref, p.g, atol=1e-4)

    # Check if remove_state_impl work correctly.
    s.clear_parameters()
