
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
3. divide the result by the number of devices, :math:`N`.

That is the underlying foundation of Data Parallel Distributed Training.

This tutorial shows the usage of Multi Process Data Parallel
Communicator for data parallel distributed training with a very simple
example.

NOTE
~~~~

This tutorial depends on **IPython Cluster**, thus when you want to run
the following excerpts of the scripts on Jupyter Notebook, follow
**`this <https://ipython.org/ipython-doc/3/parallel/parallel_process.html#using-ipcluster-in-mpiexec-mpirun-mode>`__**
to enable mpiexec/mpirun mode, then launch a corresponding Ipython
Cluster on Ipython Clusters tab.

Launch client
-------------

This codes are **only** needed for this turoial on **Jupyter Notebook**.

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

    %%px
    extension_module = "cuda.cudnn"
    ctx = extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = extension_context(extension_module, device_id=device_id)

Check different ranks are assigned to different devices

.. code:: python

    %%px
    print("n_devices={}".format(n_devices))
    print("mpi_rank={}".format(mpi_rank))


.. parsed-literal::

    [stdout:0] 
    n_devices=2
    mpi_rank=0
    [stdout:1] 
    n_devices=2
    mpi_rank=1


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

Add trainable parameters and create a solver.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %%px
    # Add parameters to communicator
    comm.add_context_and_parameters((ctx, nn.get_parameters()))
    
    # Solver and add parameters
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())

Training
~~~~~~~~

Recall the basic usage of ``nnabla`` API for training a neural netwrok,
it is

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. solver.update()

In use of ``C.MultiProcessDataParalellCommunicator``, these steps are performed in
different GPUs, and the **only difference** from these steps is
``comm.allreduce()`` Thus, in case of ``C.MultiProcessDataParalellCommunicator``
training steps are as follows,

1. loss.forward()
2. solver.zero\_grad()
3. loss.backward()
4. **comm.allreduce()**
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
    ('conv/W', array([[[[ 0.06888472,  0.03302665,  0.00224538],
             [ 0.10095084,  0.36394489,  0.00659006],
             [ 0.15155329,  0.36173904,  0.20400617]]]], dtype=float32))
    ('conv/b', array([ 0.09519047], dtype=float32))
    ('affine/W', array([[ 0.23829283, -0.23829281],
           [ 0.25489166, -0.25489166],
           [ 0.07387832, -0.0738783 ],
           ..., 
           [ 0.34147066, -0.34147066],
           [ 0.33993909, -0.33993909],
           [ 0.07020829, -0.07020829]], dtype=float32))
    ('affine/b', array([ 0.18422271, -0.1842227 ], dtype=float32))
    [stdout:1] 
    ('conv/W', array([[[[ 0.28718406,  0.19707698,  0.21287963],
             [ 0.27262157,  0.48162708,  0.58341372],
             [ 0.09545794,  0.37022409,  0.39285854]]]], dtype=float32))
    ('conv/b', array([ 0.45548177], dtype=float32))
    ('affine/W', array([[ 0.19560671, -0.19560665],
           [ 0.5929324 , -0.59293228],
           [ 0.81732005, -0.81731993],
           ..., 
           [ 0.30037487, -0.30037481],
           [ 0.33988202, -0.33988199],
           [ 0.1787488 , -0.1787488 ]], dtype=float32))
    ('affine/b', array([ 0.23541948, -0.23541945], dtype=float32))


You can see the different values on each device.

.. code:: python

    %%px
    comm.allreduce(division=True)

Commonly, ``allreduce`` only means the sum; however, ``comm.allreduce``
addresses both cases: summation and summation division.

Check gradients of weights again,

.. code:: python

    %%px
    for n, v in nn.get_parameters().items():
        print(n, v.g)


.. parsed-literal::

    [stdout:0] 
    ('conv/W', array([[[[ 0.17803439,  0.11505181,  0.1075625 ],
             [ 0.1867862 ,  0.422786  ,  0.29500189],
             [ 0.12350561,  0.36598158,  0.29843235]]]], dtype=float32))
    ('conv/b', array([ 0.27533612], dtype=float32))
    ('affine/W', array([[ 0.21694976, -0.21694973],
           [ 0.42391205, -0.42391199],
           [ 0.4455992 , -0.44559911],
           ..., 
           [ 0.32092276, -0.32092273],
           [ 0.33991057, -0.33991054],
           [ 0.12447855, -0.12447855]], dtype=float32))
    ('affine/b', array([ 0.20982111, -0.20982108], dtype=float32))
    [stdout:1] 
    ('conv/W', array([[[[ 0.17803439,  0.11505181,  0.1075625 ],
             [ 0.1867862 ,  0.422786  ,  0.29500189],
             [ 0.12350561,  0.36598158,  0.29843235]]]], dtype=float32))
    ('conv/b', array([ 0.27533612], dtype=float32))
    ('affine/W', array([[ 0.21694976, -0.21694973],
           [ 0.42391205, -0.42391199],
           [ 0.4455992 , -0.44559911],
           ..., 
           [ 0.32092276, -0.32092273],
           [ 0.33991057, -0.33991054],
           [ 0.12447855, -0.12447855]], dtype=float32))
    ('affine/b', array([ 0.20982111, -0.20982108], dtype=float32))


You can see the same values over the devices because of ``allreuce``.

Update weights,

.. code:: python

    %%px
    solver.update()

That's all for the usage of ``C.MultiProcessDataParalellCommunicator`` in the
sense of Data Parallel Distributed Training.

Now you got the picture of using ``C.MultiProcessDataParalellCommunicator``, go to
the cifar10 example,

1. **multi\_device\_multi\_process\_classification.sh**
2. **multi\_device\_multi\_process\_classification.py**

for more details.

