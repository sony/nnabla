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

import numpy as np
from collections import OrderedDict

import nnabla as nn
from nnabla.function import PythonFunction

from .backward_functions import mappings


class GradEndFunction(PythonFunction):
    """This nnabla function is used for the very end of a computational graph in the grad function.
    """

    @property
    def name(self):
        return 'GradEndFunction'

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        i = inputs[0]
        o = outputs[0]
        o.reset_shape(i.shape, True)

    def forward_impl(self, inputs, outputs):
        o = outputs[0].data.fill(1.0)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        gi = inputs[0].grad.fill(0.0)


class Grad(object):

    def __init__(self, ):
        pass

    def _force_list(self, x):
        if isinstance(x, list):
            return x
        elif hasattr(x, "__iter__"):
            return [o for o in x]
        else:
            return [x]

    def _connect_on_backward_graph(self, grad_vars, f):
        # 1. accumulate variables used more than one or do nothing
        vf_vb_map = grad_vars.pop(f)  # {VO_fwd: [VI_bwd]}
        grad_inputs = []
        for v in vf_vb_map.values():
            if len(v) > 1:
                grad_inputs += [sum(v)]
            else:
                grad_inputs += v

        # 2. check if grad depends on output data
        f_outputs = []
        for i in range(len(f.inputs)):
            for j in range(len(f.outputs)):
                if f.grad_depends_output_data(i, j):
                    # overhead for putting all outputs, but there is no way
                    f_outputs += f.outputs
                    break
        # 3. instantiate backward class
        f_fwd_name = f.info.type_name
        if f_fwd_name not in mappings:
            raise ValueError(
                "{} is not in the backward function mappings".format(f_fwd_name))
        f_bwd_class = mappings[f_fwd_name]
        f_bwd = f_bwd_class(f.context)
        f_bwd.set_num_inputs_and_outputs(len(f.inputs) + len(f_outputs) + len(grad_inputs),
                                         len(f.inputs),
                                         len(f.inputs),
                                         len(f.outputs))
        f_bwd.set_forward_function(f)

        # 4. connect
        grad_inputs = f.inputs + f_outputs + grad_inputs
        grad_outputs = f_bwd(*grad_inputs)
        grad_outputs = self._force_list(grad_outputs)

        # 5. put each grad_output as grad_input to a corresponding forward function
        for inp, grad_out in zip(f.inputs, grad_outputs):
            if inp.parent not in grad_vars:
                grad_vars[inp.parent] = OrderedDict()
            if inp not in grad_vars[inp.parent]:
                grad_vars[inp.parent][inp] = [grad_out]
            else:
                grad_vars[inp.parent][inp] += [grad_out]

        return grad_outputs

    def __call__(self, outputs, inputs, grad_outputs=None,
                 persistent_outputs=[], bind_grad_output=False):
        """
        The logic of this method is almost same as one in visit_function_backward in C++ layer.
        """
        # TODO: address test in the dynamic graph mode
        # TODO: address inplace-Function and its test
        # TODO: address auto_forward is very slow. It may be python overhead since small diff when BS is large.
        # TODO: address auto_forward consumes lots of memory, need to call v.get_unlinked_variable()?
        # TODO: address auto_forward consumes lots of memory, need to use NdArray as inputs?
        # TODO: address `set default context`

        # Check outputs/inputs
        outputs = self._force_list(outputs)
        if not all([isinstance(o, nn.Variable) for o in outputs]):
            raise ValueError("Element of outputs must be `nnabla.Variable`.")
        inputs = self._force_list(inputs)
        if not all([isinstance(i, nn.Variable) for i in inputs]):
            raise ValueError("Element of inputs must be `nnabla.Variable`.")

        # Check grad_outputs
        if grad_outputs is None:
            grad_outputs = [None] * len(outputs)
        elif isinstance(grad_outputs, (int, float, np.ndarray, nn.NdArray)):
            grad_outputs = self._force_list(grad_outputs)
        elif isinstance(grad_outputs, list):
            if len(outputs) != len(grad_outputs):
                raise ValueError(
                    "Length of `grad_outputs` and length of `outputs` must be the same.")
            for i in range(len(outputs)):
                o = outputs[i]
                go = grad_outputs[i]
                if not isinstance(go, (type(None), int, float, np.ndarray, nn.NdArray)):
                    raise ValueError("Element of `grad_outputs` must be "
                                     "in (`None`, `int`, `float`, `numpy.ndarray`, `nnabla.NdArray`) or "
                                     "list of (`None`, `int`, `float`, `numpy.ndarray`, `nnabla.NdArray`)\n"
                                     "type(grad_outputs[{}] = {}".format(i, type(go)))
                elif isinstance(go, np.ndarray) and go.shape != o.shape:
                    raise ValueError("Shape of each of outputs and grad_outputs must be same.\n"
                                     "output[{}]({}) != grad_output[{}]({})".format(i, o.shape, i, go.shape))
                elif isinstance(go, nn.NdArray) and go.shape != o.shape:
                    raise ValueError("Shape of each of outputs and grad_outputs must be same.\n"
                                     "output[{}]({}) != grad_output[{}]({})".format(i, o.shape, i, go.shape))
        # Check persistent_outputs
        if len(persistent_outputs) != 0 and len(outputs) != len(persistent_outputs):
            raise ValueError("Length of outputs and persistent_outputs "
                             "must be the same except for "
                             "the case that the length of the persistent_outputs is 0.")

        # Persistent outputs since outputs are basically losses to be monitored
        persistent_outputs = [
            True] * len(outputs) if persistent_outputs == [] else persistent_outputs
        for o, p in zip(outputs, persistent_outputs):
            o.persistent = p

        # Set grad_outputs
        for i in range(len(outputs)):
            o = outputs[i]
            go = grad_outputs[i]
            if go is None:
                pass
            elif isinstance(go, (int, float)):
                grad_output = nn.Variable(o.shape).apply(d=go, need_grad=False)
                outputs[i] = o * grad_output
            elif isinstance(go, np.ndarray):
                grad_output = nn.Variable(o.shape).apply(d=go, need_grad=False)
                outputs[i] = o * grad_output
            elif isinstance(go, nn.NdArray):
                grad_output = nn.Variable(o.shape).apply(
                    data=go, need_grad=False)
                outputs[i] = o * grad_output

        # Coerce to sum if there is multiple outputs
        output = sum(outputs) if len(outputs) != 1 else outputs[0]

        # Connect the forward and backward graph
        grad_output = GradEndFunction()(output).apply(need_grad=False)

        # Open list of next search candidate
        ids = {}

        def get_id(func):
            if func not in ids.keys():
                size = len(ids)
                ids[func] = size
                return size
            return ids[func]
        open = set()
        func = output.parent
        open.add((-output.rank, get_id(func), func))

        # Map for grad_variables consumed on the backward graph
        grad_vars = OrderedDict()  # {F_fwd: {VO_fwd: [VI_bwd]}}
        grad_vars[func] = OrderedDict({output: [grad_output]})

        # Return grads
        wrt_inputs = inputs
        grads = [None] * len(wrt_inputs)

        # Expand the forward graph to the backward graph
        while len(open) != 0:
            open = sorted(open)  # python set is NOT sorted set.
            rank_func = open.pop(0)  # 0 is necessary
            open = set(open)
            f = rank_func[2]
            if not f.need_grad:
                continue
            # Connect variables on the backward graph
            grad_outputs = self._connect_on_backward_graph(grad_vars, f)

            # Check grads w.r.t. inputs
            for inp, grad_out in zip(f.inputs, grad_outputs):
                if inp not in wrt_inputs or inp.need_grad == False:
                    continue
                idx = wrt_inputs.index(inp)
                if grads[idx] is None:
                    grads[idx] = grad_out
                else:
                    grads[idx] += grad_out  # accum at leaf
                if bind_grad_output:
                    inp.grad = grads[idx].data

            # Propagate down
            for inp in f.inputs:
                if not inp.need_grad:
                    continue
                p_i = inp.parent
                if not p_i:
                    continue
                open.add((-p_i.rank, get_id(p_i), p_i))

        return grads


def grad(outputs, inputs, grad_outputs=None, persistent_outputs=[], bind_grad_output=False):
    r"""Gradient function for the outputs with respect to the inputs.

    The grad function computes the sum of gradients of the outputs w.r.t. the inputs.

    .. math::

       g_i = \sum_{j} {\frac{\partial y_j}{\partial x_i}}, 

    :math:`y_j` is each output, :math:`x_i` is each input, and :math:`g_i` is the sum of the gradient of :math:`y_j` w.r.t. :math:`x_i` over all :math:`j`.

    Args: 
        outputs (list of :obj:`~nnabla.Variable` or :obj:`~nnabla.Variable`): Outputs of the differentiable function.
        inputs (list of :obj:`~nnabla.Variable` or :obj:`~nnabla.Variable`): Inputs w.r.t. which the gradients of outputs are computed.
        grad_outputs (None, scalar, :obj:`numpy.ndarray`, :obj:`nnabla.NdArray`, or list of scalar, :obj:`numpy.ndarray`, or :obj:`nnabla.NdArray`, ): Gradient outputs corresponding to outputs. This is same as the grad argument of :meth:`~nnabla.Variable.backward`. Default is None, so 1 is used as the in-coming gradient at the very beginning of the Variable in the backward graph.
        persistent_outputs (list of `bool`): Outputs become persistent accordingly. If not specified, all outputs become persistent.
        bind_grad_output (`bool`): Bind data to grad of input variable. This is useful for the case where one wants to use the backward graph for training a neural network using the first-order gradients only. Default is False.

    Returns
        List of :obj:`~nnabla.Variable`s. 

        If the backpropagation does not reach input(s), the corresponding returned value(s) are None.

    Example: 

    .. code-block:: python

        import nnabla as nn
        import nnabla.functions as F
        import nnabla.parametric_functions as PF
        import numpy as np
        from nnabla.ext_utils import get_extension_context

        # Context
        extension_module = "cudnn"
        ctx = get_extension_context(extension_module)

        # Input and label
        x = nn.Variable.from_numpy_array(np.random.randn(4, 3, 32, 32))
        y = nn.Variable.from_numpy_array(np.random.randint(0, 10, 4).reshape(4, 1))

        # Network
        h = PF.convolution(x, 8, (3, 3), (1, 1), name="conv1")
        h = F.relu(h)
        h = F.max_pooling(h, (2, 2))
        h = PF.convolution(h, 16, (3, 3), (1, 1), name="conv2")
        h = F.relu(h)
        h = F.max_pooling(h, (2, 2))
        p = PF.affine(h, 10, name="pred")
        loss = F.mean(F.softmax_cross_entropy(p, y))

        # Grad
        outputs = [loss]
        inputs = nn.get_parameters().values()
        grads = nn.grad(outputs, inputs)  # gradients of the parameters

        # Double backward of the outputs w.r.t. the parameters by constraining the gradient norms.
        gp = sum([F.sum(g ** 2.0) ** 0.5 for g in grads])
        loss += gp
        loss.forward()
        loss.backward()

    """

    grad_outputs = Grad()(outputs, inputs, grad_outputs=grad_outputs,
                          persistent_outputs=persistent_outputs,
                          bind_grad_output=bind_grad_output)
    return grad_outputs
