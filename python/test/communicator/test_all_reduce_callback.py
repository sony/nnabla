# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import pytest
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("pack_size", [2, 8])
@pytest.mark.parametrize("division", [False, True])
def test_all_reduce_callback(seed, pack_size, division, comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    nn.set_default_context(comm_nccl_opts.ctx)

    nn.clear_parameters()
    x_data_list = []
    num_layers = 20
    rng = np.random.RandomState(seed)
    for l in range(num_layers):
        x_data = rng.rand(3, 4)
        x_data_list.append(x_data)

    # all_reduce_callback
    x_list1 = []
    n1 = nn.Variable([3, 4])
    n1.d = 0
    for l in range(num_layers):
        x = nn.Variable([3, 4], need_grad=True)
        n1 = F.add2(n1, x)
        x.d = x_data_list[l] * (device_id + 1)
        x.g = 0
        x_list1.append(x)
    n1.backward(clear_buffer=True,
                communicator_callbacks=comm.all_reduce_callback([v.grad for v in x_list1], pack_size, division))

    # Ref AllReduce
    x_list2 = []
    n2 = nn.Variable([3, 4])
    n2.d = 0
    for l in range(num_layers):
        x = nn.Variable([3, 4], need_grad=True)
        n2 = F.add2(n2, x)
        x.d = x_data_list[l] * (device_id + 1)
        x.g = 0
        x_list2.append(x)
    n2.backward(clear_buffer=True)
    comm.all_reduce([v.grad for v in x_list2],
                    inplace=False, division=division)

    # Check
    for x, ref in zip(x_list1, x_list2):
        assert_allclose(x.g, ref.g)


@pytest.mark.parametrize("scale_grad", [1, 0.1])
@pytest.mark.parametrize("division", [False, True])
@pytest.mark.parametrize("keep_dtype", [False, True])
def test_all_reduce_callback_narrow(comm_nccl_opts, scale_grad, division, keep_dtype):
    from nnabla.function import PythonFunction

    class ConstantGrad(PythonFunction):
        def __init__(self, ctx, value, dtype):
            super(ConstantGrad, self).__init__(ctx)
            self.value = value
            self.dtype = dtype

        @property
        def name(self):
            return 'ConstantGrad'

        def min_outputs(self):
            return 1

        def grad_depends_output_data(self, i, o):
            return False

        def grad_depends_input_data(self, i, j):
            return False

        def setup_impl(self, inputs, outputs):
            assert len(inputs) == 2
            outputs[0].reset_shape(inputs[0].shape, True)

        def forward_impl(self, inputs, outputs):
            outputs[0].data.fill(0)
            outputs[0].data.cast(self.dtype, self.ctx)

        def backward_impl(self, inputs, outputs, propagate_down, accum):
            x0, x1 = inputs
            x0.grad.fill(self.value)
            x0.grad.cast(self.dtype, self.ctx)
            x1.grad.fill(self.value)
            x1.grad.cast(self.dtype, self.ctx)

    def constant_grad(a, b, value, dtype):
        c = ConstantGrad(comm_nccl_opts.ctx, value, dtype)
        return c(a, b)

    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    nn.set_default_context(comm_nccl_opts.ctx)

    x = nn.Variable((4,))
    a = nn.Variable((2, 3, 4), need_grad=True)
    b = nn.Variable((5, 1), need_grad=True)
    c = nn.Variable(tuple(), need_grad=True)
    d = nn.Variable((1, 3), need_grad=True)

    import nnabla.solvers as S
    solver = S.Sgd()
    param_dict = dict(a=a, b=b, c=c, d=d)
    grad_list = list(v.grad for v in param_dict.values())
    solver.set_parameters(param_dict)
    dtype_list = [np.float32, np.float16, np.float32, np.float16]
    h = constant_grad(x, a, 0.1 * (comm.rank + 1), np.float32)
    h = constant_grad(h, b, 0.2 * (comm.rank + 1), np.float16)
    h = constant_grad(h, c, 0.3 * (comm.rank + 1), np.float32)
    y = constant_grad(h, d, 0.4 * (comm.rank + 1), np.float16)
    y.forward()

    # 1. standard all-reduce
    y.backward()
    solver.scale_grad(scale_grad)
    comm.all_reduce(grad_list, inplace=True, division=division)
    ref_list = [g.get_data('r').copy() for g in grad_list]

    # 2. all-reduce with callback
    pack_size = 8
    cb = comm.all_reduce_callback(
        grad_list, pack_size, division=division, scale_grad=scale_grad, keep_dtype=keep_dtype)
    y.backward(communicator_callbacks=cb)
    out_list = [g.get_data('r').copy() for g in grad_list]
    for ref, out, dt in zip(ref_list, out_list, dtype_list):
        if out.dtype == np.float16:
            ref = ref.astype(np.float16)
        assert_allclose(out, ref)
        if keep_dtype:
            assert out.dtype == dt, "Data types must be kept."
        else:
            assert out.dtype == ref.dtype, "Data type must be the same as dtype of communicator's output."

    # 3. all-reduce with callback and narrow
    ac = nn.NdArray((a.size + c.size,))
    bd = nn.NdArray((b.size + d.size,))
    a.grad = ac.narrow(0, 0, a.size).view(a.shape)
    b.grad = bd.narrow(0, 0, b.size).view(b.shape)
    c.grad = ac.narrow(0, a.size, c.size).view(c.shape)
    d.grad = bd.narrow(0, b.size, d.size).view(d.shape)
    grad_list = list(v.grad for v in param_dict.values())
    cb = comm.all_reduce_callback(
        grad_list, pack_size, division=division, scale_grad=scale_grad, keep_dtype=keep_dtype)
    if keep_dtype:
        y.backward(communicator_callbacks=cb)
        out_list = [g.get_data('r').copy() for g in grad_list]
        for ref, out, dt in zip(ref_list, out_list, dtype_list):
            if out.dtype == np.float16:
                ref = ref.astype(np.float16)
            assert_allclose(out, ref)
            assert out.dtype == dt, "Data types must be kept."
    else:
        with pytest.raises(RuntimeError):
            # Type casting of narrowed arrays is prohibited.
            y.backward(communicator_callbacks=cb)
