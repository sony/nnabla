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

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as cpp_bool
from libc.stdint cimport int64_t

cimport solver
from solver cimport CSolver
cimport _variable
from _variable cimport Variable as _Variable, CVariable

# Numpy
import numpy as np
cimport numpy as np
np.import_array()


cdef class Solver:
    """Solver interface class.

    The same API provided in this class can be used to implement various types of solvers.

    Example:

    .. code-block:: python

        # Network building comes above
        import nnabla.solvers as S
        solver = S.Sgd(lr=1e-3)
        solver.set_parameters(nn.get_parameters())

        for itr in range(num_itr):
            x.d = ... # set data
            t.d = ... # set label
            loss.forward()
            solver.zero_grad()  # All gradient buffer being 0
            loss.backward()
            solver.weight_decay(decay_rate)  # Apply weight decay
            solver.update()  # updating parameters

    Note:
        All solvers provided by NNabla belong to an inherited class of ``Solver`` . A solver is never instantiated by this class itself.

    """

    @staticmethod
    cdef create(shared_ptr[CSolver] solver, info):
        s = Solver(info)
        s.info = info
        s.solver = solver
        s.solverp = solver.get()
        return s

    def setup(self, params):
        """
        Deprecated. Call ``set_parameters`` with ``param_dict`` .
        """
        raise RuntimeError("Deprecated. Call set_parameters(param_dict).")

    def set_parameters(self, param_dict, cpp_bool reset=True, cpp_bool retain_state=False):
        """
        Set parameters by dictionary of keys and parameter Variables.

        Args:
            param_dict (dict) : key:string, value: Variable.
            reset (bool): If true, clear all parameters before setting parameters. If false,
                    parameters are overwritten or added (if it's new).
            retain_state (bool): The value is only considered if reset is false. If true and
                           a key already exists (overwriting), a state (such as momentum)
                           associated with the key will be kept if the shape of the parameter
                           and that of the new param match.
        """
        cdef vector[pair[string, shared_ptr[CVariable]]] cparams
        cdef _Variable x
        cdef string key
        cdef int i
        for key, x in param_dict.iteritems():
            cparams.push_back(pair[string, shared_ptr[CVariable]](key, (< _Variable > x).varp.variable()))
        self.solverp.set_parameters(cparams, reset, retain_state)

    def remove_parameters(self, vector[string] keys):
        """
        Remove previously registered parameters, specified by a ``vector`` of its keys.
        """
        self.solverp.remove_parameters(keys)

    def clear_parameters(self):
        """
        Clear all registered parameters and states.
        """
        self.solverp.clear_parameters()

    def set_learning_rate(self, learning_rate):
        """
        Set the learning rate.
        """
        self.solverp.set_learning_rate(learning_rate)

    def zero_grad(self):
        """
        Initialize gradients of all registered parameter by zero.
        """
        self.solverp.zero_grad()

    def update(self):
        """
        When this function is called, parameter values are updated using the gradients accumulated in backpropagation,
        stored in the ``grad`` field of the parameter ``Variable`` s.
        Update rules are implemented in the C++ core,
        in derived classes of Solver. The updated parameter values will be stored into the data field of
        the parameter ``Variable`` s.
        """
        self.solverp.update()

    def weight_decay(self, float decay_rate):
        """
        Apply weight decay to gradients.
        When called, the gradient weight will be decayed by a rate of the
        current parameter value.

        Args:
            decay_rate (float): The coefficient of weight decay.
        """
        self.solverp.weight_decay(decay_rate)

    @property
    def name(self):
        """
        Get the name of the solver.
        """
        return self.solverp.name()

    def learning_rate(self):
        """
        Get the learning rate.
        """
        return self.solverp.learning_rate()


def solver_api(func):
    """
    Decorator function which passes a current context to 1st argument of
    a decocorated function.

    Args:
        func (function): 1st argument must be a Context.

    """
    import context
    import re
    import inspect
    doc = func.__doc__
    # Parsing signature
    head = doc.splitlines()[0]
    # Extract function name and args
    m = re.match(r'([A-Za-z]\w+)\((.*)\)', head)
    name, rawargs = m.groups()
    # Parse arguments. Note: Here we drop the first argument which is a
    # context.
    r = re.compile(r'(\w+\s|)(\w+)(|=.*)$')
    parsed_args = map(
        lambda x: r.match(x.strip()).groups(), rawargs.split(',')[1:])
    types, args, defaults = zip(*parsed_args)
    types = map(lambda x: x.strip(), types)
    defaults = map(lambda x: eval(x), filter(
        lambda x: len(x) > 0, map(lambda x: x.strip(' ='), defaults)))
    # Create argument code.
    argspec = inspect.formatargspec(args, None, None, defaults)
    shortargspec = inspect.formatargspec(
        args, None, None, None)[1:-1]  # Remove ()
    code = """
def {name}{argspec}:
    ctx = context.get_current_context()
    return _func_(ctx, {shortargspec})
    """.format(**locals())
    execdict = dict(_func_=func, context=context)
    exec(code, execdict)
    func_with_context = execdict[name]
    doc = '\n'.join(doc.splitlines()[1:])
    returns_embed = """
    Returns:
        ~nnabla.solver.Solver: An intance of Solver class.
            See Solver API guide for details.

    """
    doc = doc.replace('<##Returns##>', returns_embed)
    doc += """
    Note:
        You can instantiate a preferred target implementation (ex. CUDA) of
        a Solver given a `Context`. A `Context` can be set by
        ``nnabla.set_default_context(ctx)``
        or ``nnabla.context_scope(ctx)``. See API docs.

    """
    func_with_context.__doc__ = doc
    func_with_context.__solver_api_base__ = func
    func_with_context.__module__ = __name__
    return func_with_context


@solver_api
def Adadelta(CContext ctx, float lr=1.0,
             float decay=0.95, float eps=1e-6):
    r"""
    AdaDelta optimizer.

    .. math::
        g_t &\leftarrow \Delta w_t\\
        v_t &\leftarrow - \frac{RMS \left[ v_t \right]_{t-1}}
            {RMS \left[ g \right]_t}g_t\\
        w_{t+1} &\leftarrow w_t + \eta v_t

    Args:
        lr (float): Learning rate (:math:`\eta`).
        decay (float): Decay rate (:math:`\gamma`).
        eps (float): Small value for avoiding zero devision(:math:`\epsilon`).

    <##Returns##>

    References:
        * `Matthew D. Zeiler.
          ADADELTA: An Adaptive Learning Rate Method.
          <https://arxiv.org/abs/1212.5701>`_

    """
    info = {}
    info['lr'] = lr
    info['decay'] = decay
    info['eps'] = eps
    return Solver.create(create_AdadeltaSolver(ctx, lr, decay, eps), info)


@solver_api
def Adagrad(CContext ctx, float lr=0.01, float eps=1e-8):
    r"""
    ADAGrad optimizer.

    .. math::
        g_t &\leftarrow \Delta w_t\\
        G_t &\leftarrow G_{t-1} + g_t^2\\
        w_{t+1} &\leftarrow w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t

    Args:
        lr (float): Learning rate
        eps (float): Small value for avoiding zero devision.
        lr (float): Learning rate (:math:`\eta`).
        eps (float): Small value for avoiding zero devision(:math:`\epsilon`).

    <##Returns##>

    References:
        * `John Duchi, Elad Hazan and Yoram Singer (2011).
          Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
          <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_

    """
    info = {}
    info['lr'] = lr
    info['eps'] = eps
    return Solver.create(create_AdagradSolver(ctx, lr, eps), info)


@solver_api
def Adam(CContext ctx, float alpha=1e-3,
         float beta1=0.9, float beta2=0.999, float eps=1e-8):
    r"""
    ADAM optimizer.

    .. math::
        m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
        v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
        w_{t+1} &\leftarrow w_t - \alpha
            \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
            \frac{m_t}{\sqrt{v_t} + \epsilon}

    where :math:`g_t` denotes a gradient, and let :math:`m_0 \leftarrow 0`
    and :math:`v_0 \leftarrow 0`.

    Args:
        alpha (float): Step size (:math:`\alpha`)
        beta1 (float): Decay rate of first-order momentum (:math:`\beta_1`).
        beta2 (float): Decay rate of second-order momentum (:math:`\beta_2`).
        eps (float): Small value for avoiding zero devision (:math:`\epsilon`).

    <##Returns##>

    References:
        * `Kingma and Ba, Adam: A Method for Stochastic Optimization.
          <https://arxiv.org/abs/1412.6980>`_

    """

    info = {}
    info['alpha'] = alpha
    info['beta1'] = beta1
    info['beta2'] = beta2
    info['eps'] = eps
    return Solver.create(create_AdamSolver(ctx, alpha, beta1, beta2, eps), info)


@solver_api
def Adamax(CContext ctx, float alpha=2e-3,
           float beta1=0.9, float beta2=0.999, float eps=1e-8):
    r"""
    ADAMAX Optimizer.

    .. math::
        m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
        v_t &\leftarrow \max\left(\beta_2 v_{t-1}, |g_t|\right)\\
        w_{t+1} &\leftarrow w_t - \alpha
            \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
            \frac{m_t}{v_t + \epsilon}

    where :math:`g_t` denotes a gradient, and let :math:`m_0 \leftarrow 0`
    and :math:`v_0 \leftarrow 0`, :math:`v_t` is an
    exponentially weighted infinity norm of a sequence of
    gradients :math:`t=0,...,t`.

    Args:
        alpha (float): Step size (:math:`\alpha`)
        beta1 (float): Decay rate of first-order momentum (:math:`\beta_1`).
        beta2 (float): Decay rate of inf-order momentum (:math:`\beta_2`).
        eps (float): Small value for avoiding zero devision (:math:`\epsilon`)n.

    <##Returns##>

    References:
        * `Kingma and Ba, Adam: A Method for Stochastic Optimization.
          <https://arxiv.org/abs/1412.6980>`_
    """

    info = {}
    info['alpha'] = alpha
    info['beta1'] = beta1
    info['beta2'] = beta2
    info['eps'] = eps
    return Solver.create(create_AdamaxSolver(ctx, alpha, beta1, beta2, eps), info)


@solver_api
def Momentum(CContext ctx, float lr, float momentum=0.9):
    r"""
    SGD with Momentum.

    .. math::
        v_t &\leftarrow \gamma v_{t-1} + \eta \Delta w_t\\
        w_{t+1} &\leftarrow w_t - v_t

    Args:
        lr (float): Learning rate (:math:`\eta`).
        momentum (float): Decay rate of momentum (:math:`\gamma`).

    <##Returns##>

    References:

        * `Ning Qian : On the Momentum Term in Gradient Descent Learning Algorithms.
          <http://www.columbia.edu/~nq6/publications/momentum.pdf>`_

    """
    info = {}
    info['lr'] = lr
    info['momentum'] = momentum
    return Solver.create(create_MomentumSolver(ctx, lr, momentum), info)


@solver_api
def Nesterov(CContext ctx, float lr, float momentum=0.9):
    r"""
    Nesterov Accellerated Gradient optimizer.

    .. math::
        v_t &\leftarrow \gamma v_{t-1} - \eta \Delta w_t\\
        w_{t+1} &\leftarrow w_t - \gamma v_{t-1} + \left(1 + \gamma \right) v_t

    Args:
        lr (float): Learning rate (:math:`\eta`).
        momentum (float): Decay rate of momentum (:math:`\gamma`).

    <##Returns##>

    References:
        * Yurii Nesterov.
          A method for unconstrained convex minimization problem with the rate of
          convergence :math:`o(1/k2)`.

        lr (float): Learning rate.
        momentum (float): Decay rate of momentum.
   """
    info = {}
    info['lr'] = lr
    info['momentum'] = momentum
    return Solver.create(create_NesterovSolver(ctx, lr, momentum), info)


@solver_api
def RMSprop(CContext ctx, float lr=0.001,
            float decay=0.9, float eps=1e-8):
    r"""
    RMSprop optimizeer (Geoffery Hinton).

    .. math::
        g_t &\leftarrow \Delta w_t\\
        v_t &\leftarrow \gamma v_{t-1} + \left(1 - \gamma \right) g_t^2\\
        w_{t+1} &\leftarrow w_t - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}

    Args:
        lr (float): Learning rate (:math:`\eta`).
        decay (float): Decay rate (:math:`\gamma`).
        eps (float): Small value for avoiding zero devision(:math:`\epsilon`).

    <##Returns##>

    References:
        * `Geoff Hinton.
          Lecture 6a : Overview of mini-batch gradient descent.
          <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_

    """
    info = {}
    info['lr'] = lr
    info['decay'] = decay
    info['eps'] = eps
    return Solver.create(create_RMSpropSolver(ctx, lr, decay, eps), info)


@solver_api
def Sgd(CContext ctx, float lr):
    r"""
    Stochastic gradient descent (SGD) optimizer.

    .. math::
        w_{t+1} \leftarrow w_t - \eta \Delta w_t

    Args:
        lr (float): Learning rate (:math:`\eta`).

    <##Returns##>

    """
    info = {}
    info['lr'] = lr
    return Solver.create(create_SgdSolver(ctx, lr), info)
