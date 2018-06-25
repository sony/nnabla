
Data Parallel Distributed Training
==================================

DataParallelCommunicator enables to train your neural network using
multiple devices. It is normally used for gradients exchange in data
parallel distributed training. Basically, there are two types of
distributed trainings in Neural Network literature: Data Parallel and
Model Parallel. Here we only focus on the former, Data Parallel
Training. Data Parallel Distributed Training is based on the very simple
equation used for the optimization of a neural network called
(Mini-Batch) Stochastic Gradient Descent.

In the optimization process, the objective one tries to minimize is

.. math::


   f(\mathbf{w}; X) = \frac{1}{B \times N} \sum_{i=1}^{B \times N} \ell(\mathbf{w}, \mathbf{x}_i),

where :math:`f` is a neural network, :math:`B \times N` is the batch
size, :math:`\ell` is a loss function for each data point
:math:`\mathbf{x} \in X`, and :math:`\mathbf{w}` is the trainable
parameter of the neural network.

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

In data parallel distributed training, the following steps are performed
according to the above equation,

1. each term, summation of derivatives (gradients) divided by batch size
   :math:`B`, is computed on a separated device (typically GPU),
2. take the sum over devices,
3. divide the result by the number of devices, :math:`N`.

That is the underlying foundation of Data Parallel Distributed Training.

This tutorial shows the usage of Multi Process Data Parallel
Communicator for data parallel distributed training with a very simple
example.

NOTE
~~~~

This tutorial depends on **IPython Cluster**, thus when you want to run
the following excerpts of the scripts on Jupyter Notebook, follow
`this <https://ipython.org/ipython-doc/3/parallel/parallel_process.html#using-ipcluster-in-mpiexec-mpirun-mode>`__
to enable mpiexec/mpirun mode, then launch a corresponding Ipython
Cluster on Ipython Clusters tab.

Launch client
-------------

This code is **only** needed for this tutorial via **Jupyter Notebook**.

.. code:: python

    import ipyparallel as ipp
    rc = ipp.Client(profile='mpi')

Prepare the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %%px
    import os
    import time
    
    import nnabla as nn
    import nnabla.communicators as C
    from nnabla.ext_utils import get_extension_context
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

    %%px
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = get_extension_context(extension_module, device_id=device_id)

Check different ranks are assigned to different devices

.. code:: python

    %%px
    print("n_devices={}".format(n_devices))
    print("mpi_rank={}".format(mpi_rank))


.. parsed-literal::

    [stdout:0] 
    n_devices=2
    mpi_rank=1
    [stdout:1] 
    n_devices=2
    mpi_rank=0


Create data points and a very simple neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %%px
    # Data points setting
    n_class = 2
    b, c, h, w = 4, 1, 32, 32 
    
    # Data points
    x_data = np.random.rand(b, c, h, w)
    y_data = np.random.choice(n_class, b).reshape((b, 1))
    x = nn.Variable(x_data.shape)
    y = nn.Variable(y_data.shape)
    x.d = x_data
    y.d = y_data
    
    # Network setting
    C = 1
    kernel = (3, 3)
    pad = (1, 1)
    stride = (1, 1)


.. code:: python

    %%px
    rng = np.random.RandomState(0)
    w_init = UniformInitializer(
                        calc_uniform_lim_glorot(C, C/2, kernel=(1, 1)), 
                        rng=rng)


.. code:: python

    %%px
    # Network
    with nn.context_scope(ctx):
        h = PF.convolution(x, C, kernel, pad, stride, w_init=w_init)
        pred = PF.affine(h, n_class, w_init=w_init)
        loss = F.mean(F.softmax_cross_entropy(pred, y))

**Important notice** here is that ``w_init`` is passed to parametric
functions to let the network on each GPU start from the same values of
trainable parameters in the optimization process.

Create a solver.
~~~~~~~~~~~~~~~~

.. code:: python

    %%px
    # Solver and add parameters
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

Training
~~~~~~~~

Recall the basic usage of ``nnabla`` API for training a neural network,
it is

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. solver.update()

In use of ``C.MultiProcessDataParalellCommunicator``, these steps are
performed in different GPUs, and the **only difference** from these
steps is ``comm.all_reduce()``. Thus, in case of
``C.MultiProcessDataParalellCommunicator`` training steps are as
follows,

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. **comm.all\_reduce([x.grad for x in nn.get\_parameters().values()])**
5. solver.update()

First, forward, zero\_grad, and backward,

.. code:: python

    %%px
    # Training steps
    loss.forward()
    solver.zero_grad()
    loss.backward()

Check gradients of weights once,

.. code:: python

    %%px
    for n, v in nn.get_parameters().items():
        print(n, v.g)


.. parsed-literal::

    [stdout:0] 
    ('conv/W', array([[[[ 5.0180483,  0.457942 , -2.8701296],
             [ 2.0715926,  3.0698593, -1.6650047],
             [-2.5591214,  6.4248834,  9.881935 ]]]], dtype=float32))
    ('conv/b', array([8.658947], dtype=float32))
    ('affine/W', array([[-0.93160367,  0.9316036 ],
           [-1.376812  ,  1.376812  ],
           [-1.8957546 ,  1.8957543 ],
           ...,
           [-0.33000934,  0.33000934],
           [-0.7211893 ,  0.72118926],
           [-0.25237036,  0.25237036]], dtype=float32))
    ('affine/b', array([-0.48865744,  0.48865741], dtype=float32))
    [stdout:1] 
    ('conv/W', array([[[[ -1.2505884 ,  -0.87151337,  -8.685524  ],
             [ 10.738419  ,  14.676786  ,   7.483423  ],
             [  5.612471  , -12.880402  ,  19.141157  ]]]], dtype=float32))
    ('conv/b', array([13.196114], dtype=float32))
    ('affine/W', array([[-1.6865108 ,  1.6865108 ],
           [-0.938529  ,  0.938529  ],
           [-1.028422  ,  1.028422  ],
           ...,
           [-0.98217344,  0.98217344],
           [-0.97528917,  0.97528917],
           [-0.413546  ,  0.413546  ]], dtype=float32))
    ('affine/b', array([-0.7447065,  0.7447065], dtype=float32))


You can see the different values on each device, then call
``all_reduce``,

.. code:: python

    %%px
    comm.all_reduce([x.grad for x in nn.get_parameters().values()], division=True)

Commonly, ``all_reduce`` only means the sum; however,
``comm.all_reduce`` addresses both cases: summation and summation
division.

Again, check gradients of weights,

.. code:: python

    %%px
    for n, v in nn.get_parameters().items():
        print(n, v.g)


.. parsed-literal::

    [stdout:0] 
    ('conv/W', array([[[[ 1.8837299 , -0.20678568, -5.777827  ],
             [ 6.4050055 ,  8.8733225 ,  2.9092093 ],
             [ 1.5266749 , -3.2277591 , 14.511546  ]]]], dtype=float32))
    ('conv/b', array([21.85506], dtype=float32))
    ('affine/W', array([[-2.6181145,  2.6181145],
           [-2.315341 ,  2.315341 ],
           [-2.9241767,  2.9241762],
           ...,
           [-1.3121828,  1.3121828],
           [-1.6964785,  1.6964784],
           [-0.6659163,  0.6659163]], dtype=float32))
    ('affine/b', array([-1.233364 ,  1.2333639], dtype=float32))
    [stdout:1] 
    ('conv/W', array([[[[ 1.8837299 , -0.20678568, -5.777827  ],
             [ 6.4050055 ,  8.8733225 ,  2.9092093 ],
             [ 1.5266749 , -3.2277591 , 14.511546  ]]]], dtype=float32))
    ('conv/b', array([21.85506], dtype=float32))
    ('affine/W', array([[-2.6181145,  2.6181145],
           [-2.315341 ,  2.315341 ],
           [-2.9241767,  2.9241762],
           ...,
           [-1.3121828,  1.3121828],
           [-1.6964785,  1.6964784],
           [-0.6659163,  0.6659163]], dtype=float32))
    ('affine/b', array([-1.233364 ,  1.2333639], dtype=float32))


You can see the same values over the devices because of ``all_reduce``.

Update weights,

.. code:: python

    %%px
    solver.update()

This concludes the usage of ``C.MultiProcessDataParalellCommunicator``
for Data Parallel Distributed Training.

Now you should have an understanding of how to use
``C.MultiProcessDataParalellCommunicator``, go to the cifar10 example,

1. **multi\_device\_multi\_process\_classification.sh**
2. **multi\_device\_multi\_process\_classification.py**

for more details.
