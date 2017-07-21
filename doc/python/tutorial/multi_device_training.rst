
Data Parallel Distributed Training
==================================

DataParallelCommunicator enables to train your neural network using
multiple devices. It is normally used for gradients exchange in data
parallel distributed training. Basically, there are two types of
distributed trainings in Neural Network literature: Data Parallel and
Model Parallel. Here we only focus on the former, Data Parallel
Training. Data Parallel Distributed Training are based on the very
simple equation in the optimization for a neural network called
(Mini-Batch) Stochastic Gradient Descent.

In the oprimization process, the objective one tries to minimize is

.. math::


   f(\mathbf{w}; X) = \frac{1}{B \times N} \sum_{i=1}^{B \times N} \ell(\mathbf{w}, \mathbf{x}_i),

where :math:`f` is a neural network, :math:`B \times N` is the batch
size, :math:`\ell` is a loss function for each data point
:math:`\mathbf{x} \in X`, and :math:`\mathbf{w}` is the trainable
parameter of the neural newtwork.

When taking the derivative of this objective, one gets,

.. math::


   \nabla_{\mathbf{w}} f(\mathbf{w}; X) = \frac{1}{B \times N} \sum_{i=1}^{B \times N} \nabla_{\mathbf{w}} \ell (\mathbf{w}, \mathbf{x}_i).

Since the derivative has linearity, one can change the objective to the
sum of summations each of which is the sum of derivatives over :math:`B`
data points.

.. math::


   \nabla_{\mathbf{w}} f(\mathbf{w}; X) = \frac{1}{N} \left(
    \frac{1}{B} \sum_{i=1}^{B} \nabla_{\mathbf{w}} \ell (\mathbf{w}, \mathbf{x}_i) \
    + \frac{1}{B} \sum_{i=B+1}^{B \times 2} \nabla_{\mathbf{w}} \ell (\mathbf{w}, \mathbf{x}_i) \
    + \ldots \
    + \frac{1}{B} \sum_{i=B \times (N-1) + 1}^{B \times N} \nabla_{\mathbf{w}} \ell (\mathbf{w}, \mathbf{x}_i)
   \right)

In data parallel distributed training, the follwoing steps are peformed
according to the above equation,

1. each term, summation of derivatives (gradients) divided by batch size
   :math:`B`, is computed on a separated device (tipically GPU),
2. take the sum over devices,
3. divide the result by the number of devices.

That is the underlying foundation of Data Parallel Distributed Training.

This tutorial shows the usage of Multi Process Data Parallel
Communicator for data parallel distributed training with a very simple
example.

Prepare the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import os
    import time
    
    import nnabla as nn
    import nnabla.communicators as C
    from nnabla.contrib.context import extension_context
    import nnabla.functions as F
    from nnabla.initializer import (
        calc_uniform_lim_glorot,
        UniformInitializer)
    import nnabla.parametric_functions as PF
    import nnabla.solvers as S
    import numpy as np

Define the communicator for gradients exchange.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    extension_module = "cuda.cudnn"
    ctx = extension_context(extension_module)
    comm = C.mpDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = extension_context(extension_module, device_id=device_id)

Create data points and a very simple neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Data points setting
    n_class = 10
    b, c, h, w = 8, 3, 32, 32 
    
    # Data points
    x_data = np.random.rand(b, c, h, w)
    y_data = np.random.choice(n_class, b).reshape((b, 1))
    x = nn.Variable(x_data.shape)
    y = nn.Variable(y_data.shape)
    x.d = x_data
    y.d = y_data
    
    # Network setting
    C = 16
    kernel = (3, 3)
    pad = (1, 1)
    stride = (1, 1)
    rng = np.random.RandomState(0)
    w_init = UniformInitializer(
                        calc_uniform_lim_glorot(C, C/2, kernel=(1, 1)), 
                        rng=rng)
    
    # Network
    with nn.context_scope(ctx):
        h = PF.convolution(x, C, kernel, pad, stride, w_init=w_init)
        pred = PF.affine(h, n_class)
        loss = F.mean(F.softmax_cross_entropy(pred, y))

**Important notice** here is that ``w_init`` is passed to parametric
functions to let the network on each GPU start from the same values of
trainable parameters in the optimization process.

Add trainable parameters and create a solver.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Add parameters to communicator
    comm.add_context_and_parameters((ctx, nn.get_parameters()))
    
    # Solver and add parameters
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

Recall the basic usage of ``nnabla`` API for training a neural netwrok,
it is

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. solver.update()

In use of ``C.mpDataParalellCommunicator``, these steps are performed in
different GPUs, and the **only difference** from these steps is
``comm.allreduce()`` Thus, in case of ``C.mpDataParalellCommunicator``
training steps are as follows,

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. **comm.allreduce()**
5. solver.update()

.. code:: python

    # Training steps
    loss.forward()
    solver.zero_grad()
    loss.backward()
    comm.allreduce(division=True)
    solver.update()

Commonly, ``allreduce`` only means the sum; however, ``comm.allreduce``
addresses both cases: summation and summation division.

That's all for the usage of ``C.mpDataParallelCommunicator`` in the
sense of Data Parallel Distributed Training.

Now you got the picture of using ``C.mpDataParallelCommunicator``, go to
the cifar10 example, **
multi\_device\_multi\_process\_classification.py** for more details.

