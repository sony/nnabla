# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import nnabla.functions as F
from nnabla.function import PythonFunction
from .backward_functions import registry


class GradEndFunction(PythonFunction):
    """This nnabla function is used for the very end of a computational graph in the grad function.
    """

    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 1

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False

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

    def _connect_on_gradient_graph(self, grad_vars, f):
        # 1. accumulate variables used more than one or do nothing
        vf_vb_map = grad_vars.pop(f)  # {VO_fwd: [VI_bwd]}
        grad_inputs = []
        for o in f.outputs:
            # address `floating` variables; no function takes it as input.
            # e.g., when dx, db, dg = dBN(...), (db, dg) are not used afterwards.
            # [0] is not enough, e.g., F.concatenate
            v = vf_vb_map.get(o, [F.constant(0.0, o.shape)])
            if len(v) > 1:
                grad_inputs += [sum(v)]
                # grad_inputs += [F.add_n(v)]
            else:
                grad_inputs += v

        # 2. lookup the backward function
        f_fwd_name = f.info.type_name
        if f_fwd_name not in registry:
            raise ValueError(
                "{} is not in the backward function registry".format(f_fwd_name))
        backward_func = registry[f_fwd_name]

        # 3. connect
        # Only grad-depending inputs are passed to the backward function because
        # the other inputs are probably cleared before.
        grad_depending_inputs = [None for inp in f.inputs]
        input_shapes = [inp.shape for inp in f.inputs]
        for i in range(len(f.inputs)):
            for j, inp in enumerate(f.inputs):
                if f.grad_depends_input_data(i, j) or \
                   f.auto_grad_depends_input_data(i, j):
                    grad_depending_inputs[j] = inp

        grad_depending_outputs = [None for outp in f.outputs]
        output_shapes = [outp.shape for outp in f.outputs]
        for i in range(len(f.inputs)):
            for o, outp in enumerate(f.outputs):
                if f.grad_depends_output_data(i, o) or \
                   f.auto_grad_depends_output_data(i, o):
                    grad_depending_outputs[o] = outp

        ctx = nn.get_current_context()
        with nn.context_scope(ctx):
            grad_outputs = backward_func(grad_inputs, grad_depending_inputs,
                                         input_shapes, grad_depending_outputs,
                                         output_shapes, **f.info.args)
        grad_outputs = self._force_list(grad_outputs)

        # 4. put grad_output as grad_input to a corresponding function
        for inp, grad_out in zip(f.inputs, grad_outputs):
            if grad_out is None:
                continue
            if inp.parent not in grad_vars:
                grad_vars[inp.parent] = OrderedDict()
            if inp not in grad_vars[inp.parent]:
                grad_vars[inp.parent][inp] = [grad_out]
            else:
                grad_vars[inp.parent][inp] += [grad_out]
        return grad_outputs

    def _get_corresponding_vars_on_graph(self, inputs, outputs):
        """
        This methods binds the inputs which may be disconnected from a computation graph 
        (e.g., trainable parameters obtrained by nn.get_parmaeters()) to actual inputs connected 
        to a given computation graph traversabled from outputs
        """
        ret = [None for _ in range(len(inputs))]
        inp2id = {x: i for i, x in enumerate(inputs)}

        open = set()

        def dfs(node):
            if node in open or not node.parent:
                return

            for par_var in node.parent.inputs:
                if node in open:
                    continue

                id = inp2id.get(par_var, None)
                if id is not None:
                    ret[id] = par_var

                dfs(par_var)
                open.add(par_var)

        for output in outputs:
            dfs(output)

        return ret

    def _get_children(self, wrt_inputs):
        """
        Return children traversable from `wrt_inputs` in the forward graph topological order.
        """
        children = set()

        def dfs(node):
            if node in children:
                return

            for f in node.function_references:
                if f.info.type_name == "GradEndFunction":
                    continue
                for o in f.outputs:
                    if o in children:
                        continue
                    dfs(o)
                    children.add(o)

        for inp in wrt_inputs:
            if inp is None:
                continue
            dfs(inp)

        return children

    def __call__(self, outputs, inputs, grad_outputs=None,
                 persistent_outputs=[], bind_grad_output=False):
        """
        The logic of this method is almost same as one in visit_function_backward in C++ layer.
        """
        # TODO: address auto_forward is very slow. It may be python overhead since small diff when BS is large.
        # TODO: address auto_forward consumes lots of memory, need to call v.get_unlinked_variable()?
        # TODO: address auto_forward consumes lots of memory, need to use NdArray as inputs?

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
                if not isinstance(go, (type(None), int, float, np.ndarray, nn.NdArray, nn.Variable)):
                    raise ValueError("Element of `grad_outputs` must be "
                                     "in (`None`, `int`, `float`, `numpy.ndarray`, "
                                     "`nnabla.NdArray`, `nnabla.Variable`) or "
                                     "list of (`None`, `int`, `float`, `numpy.ndarray`, "
                                     "`nnabla.NdArray`, `nnabla.Variable`)\n"
                                     "type(grad_outputs[{}] = {}".format(i, type(go)))
                elif isinstance(go, (np.ndarray, nn.NdArray, nn.Variable)) and go.shape != o.shape:
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

        # Open list of next search candidate
        ids = {}

        def get_id(func):
            if func not in ids.keys():
                size = len(ids)
                ids[func] = size
                return size
            return ids[func]
        open = set()

        # Map for grad_variables consumed on the gradient graph.
        # End is the special case where d_o = end_f(o) and map[end_f] = {o: [d_o]}
        grad_vars = OrderedDict()  # {F_fwd: {VO_fwd: [VI_bwd]}}

        # Set grad_outputs
        for i in range(len(outputs)):
            o = outputs[i]
            go = grad_outputs[i]
            if go is None:
                output = o
            elif isinstance(go, (int, float)):
                go = nn.Variable(o.shape).apply(d=go, need_grad=False)
                output = o * go
            elif isinstance(go, np.ndarray):
                go = nn.Variable(o.shape).apply(d=go, need_grad=False)
                output = o * go
            elif isinstance(go, nn.NdArray):
                go = nn.Variable(o.shape).apply(data=go, need_grad=False)
                output = o * go
            elif isinstance(go, nn.Variable):
                output = o * go
            func = output.parent
            open.add((-output.rank, get_id(func), func))

            # Connect the graph and its gradient graph
            grad_output = GradEndFunction()(output).apply(need_grad=False)
            grad_vars[func] = OrderedDict({output: [grad_output]})

        # Return grads but
        # replace inputs params with the vars connected with the given graph
        wrt_inputs = self._get_corresponding_vars_on_graph(inputs, outputs)
        grads = [None] * len(wrt_inputs)
        child_nodes = self._get_children(wrt_inputs)
        wrt_inputs = [nn.Variable() if x is None else x for x in wrt_inputs]

        # Expand the graph to its gradient graph
        while len(open) != 0:
            open = sorted(open)  # python set is NOT sorted set.
            rank_func = open.pop(0)  # 0 is necessary
            open = set(open)
            f = rank_func[2]

            if not f.need_grad:
                continue
            # Connect variables on the gradient graph
            grad_outputs = self._connect_on_gradient_graph(grad_vars, f)

            # Check grads w.r.t. inputs
            for inp, grad_out in zip(f.inputs, grad_outputs):
                if inp not in wrt_inputs or inp.need_grad == False or grad_out is None:
                    continue
                idx = wrt_inputs.index(inp)
                if grads[idx] is None:
                    grads[idx] = grad_out
                else:
                    grads[idx] = grads[idx] + grad_out  # accum at leaf
                if bind_grad_output:
                    inp.grad = grads[idx].data

            # Propagate down
            for inp, grad_out in zip(f.inputs, grad_outputs):
                if inp not in child_nodes or not inp.need_grad or grad_out is None:
                    continue
                p_i = inp.parent
                if not p_i:
                    continue
                open.add((-p_i.rank, get_id(p_i), p_i))

        # If the final grads has None, then None becomes zero Variable(s).
        for i in range(len(grads)):
            if grads[i]:
                continue
            grads[i] = F.constant(0, wrt_inputs[i].shape)
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
        grad_outputs (None, scalar, :obj:`numpy.ndarray`, :obj:`nnabla.NdArray`, or list of scalar, :obj:`numpy.ndarray`, or :obj:`nnabla.NdArray`, ): Gradient outputs corresponding to outputs. This is same as the grad argument of :meth:`~nnabla.Variable.backward`. Default is None, so 1 is used as the in-coming gradient at the very beginning of the Variable in the gradient graph.
        persistent_outputs (list of `bool`): Outputs become persistent accordingly. If not specified, all outputs become persistent.
        bind_grad_output (`bool`): Bind data to grad of input variable. This is useful for the case where one wants to use the gradient graph for training a neural network using the first-order gradients only. Default is False.

    Returns
        List of :obj:`~nnabla.Variable`.

        If the backpropagation does not reach input(s), the corresponding returned value(s) are `zero`
        (i.e., the gradients w.r.t. inputs are zero) and not connected as a part of the gradient graph.

    Example (Gradient Penalty): 

    .. code-block:: python

        import nnabla as nn
        import nnabla.functions as F
        import nnabla.parametric_functions as PF
        import numpy as np
        from nnabla.ext_utils import get_extension_context

        # Context
        extension_module = "cudnn"
        ctx = get_extension_context(extension_module)
        nn.set_default_context(ctx)

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

        # Backward of the outputs w.r.t. the parameters by constraining the gradient norms.
        t = 0 # or 1
        gp = sum([(F.sum(g ** 2) ** 0.5 - t) ** 2 for g in grads])
        loss += gp
        loss.forward()
        loss.backward()


    Example (Higer-order Gradients): 

    .. code-block:: python

        import nnabla as nn
        import nnabla.functions as F
        import numpy as np

        x = nn.Variable.from_numpy_array(np.random.randn(2, 2)).apply(need_grad=True)
        x.grad.zero()
        y = F.sin(x)
        def grad(y, x, n=1):
            dx = [y]
            for _ in range(n):
                dx = nn.grad([dx[0]], [x])
            return dx[0]
        dnx = grad(y, x, n=10)
        dnx.forward()
        print(np.allclose(-np.sin(x.d), dnx.d))
        dnx.backward()
        print(np.allclose(-np.cos(x.d), x.g))

        # Show the supported status for each function
        from nnabla.backward_functions import show_registry
        show_registry()
        """

    grad_outputs = Grad()(outputs, inputs, grad_outputs=grad_outputs,
                          persistent_outputs=persistent_outputs,
                          bind_grad_output=bind_grad_output)
    return grad_outputs
