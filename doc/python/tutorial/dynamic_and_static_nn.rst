
Static vs Dynamic Neural Networks in NNabla
===========================================

NNabla allows you to define static and dynamic neural networks. Static
neural networks have a fixed layer architecture, i.e., a static
computation graph. In contrast, dynamic neural networks use a dynamic
computation graph, e.g., randomly dropping layers for each minibatch.

This tutorial compares both computation graphs.

.. code-block:: python2

    %matplotlib inline
    import nnabla as nn
    import nnabla.functions as F
    import nnabla.parametric_functions as PF
    import nnabla.solvers as S
    
    import numpy as np
    np.random.seed(0)
    
    GPU = 0  # ID of GPU that we will use


.. parsed-literal::

    2017-06-26 23:10:05,832 [nnabla][INFO]: Initializing CPU extension...


Dataset loading
~~~~~~~~~~~~~~~

We will first setup the digits dataset from scikit-learn:

.. code-block:: python2

    from tiny_digits import *
    
    digits = load_digits()
    data = data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)


.. parsed-literal::

    2017-06-26 23:10:06,042 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:06,043 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 23:10:06,044 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:06,044 [nnabla][INFO]: On-memory
    2017-06-26 23:10:06,045 [nnabla][INFO]: Using DataIterator


Each sample in this dataset is a grayscale image of size 8x8 and belongs
to one of the ten classes ``0``, ``1``, ..., ``9``.

.. code-block:: python2

    img, label = data.next()
    print img.shape, label.shape


.. parsed-literal::

    (16, 1, 8, 8) (16, 1)


Network definition
~~~~~~~~~~~~~~~~~~

As an example, we define a (unnecessarily) deep CNN:

.. code-block:: python2

    def cnn(x):
        """Unnecessarily Deep CNN.
        
        Args:
            x : Variable, shape (B, 1, 8, 8)
            
        Returns:
            y : Variable, shape (B, 10)
        """
        with nn.parameter_scope("cnn"):  # Parameter scope can be nested
            with nn.parameter_scope("conv1"):
                h = F.tanh(PF.batch_normalization(
                    PF.convolution(x, 64, (3, 3), pad=(1, 1))))
            for i in range(10):  # unnecessarily deep
                with nn.parameter_scope("conv{}".format(i + 2)):
                    h = F.tanh(PF.batch_normalization(
                        PF.convolution(h, 128, (3, 3), pad=(1, 1))))
            with nn.parameter_scope("conv_last"):
                h = F.tanh(PF.batch_normalization(
                    PF.convolution(h, 512, (3, 3), pad=(1, 1))))
                h = F.average_pooling(h, (2, 2))
            with nn.parameter_scope("fc"):
                h = F.tanh(PF.affine(h, 1024))
            with nn.parameter_scope("classifier"):
                y = PF.affine(h, 10)
        return y

Static computation graph
------------------------

First, we will look at the case of a static computation graph where the
neural network does not change during training.

.. code-block:: python2

    from nnabla.ext_utils import get_extension_context
    
    # setup cuda extension
    ctx_cuda = get_extension_context('cudnn', device_id=GPU)  # replace 'cudnn' by 'cpu' if you want to run the example on the CPU
    nn.set_default_context(ctx_cuda)
    
    # create variables for network input and label
    x = nn.Variable(img.shape)
    t = nn.Variable(label.shape)
    
    # create network
    static_y = cnn(x)
    static_y.persistent = True
    
    # define loss function for training
    static_l = F.mean(F.softmax_cross_entropy(static_y, t))


.. parsed-literal::

    2017-06-26 23:10:06,350 [nnabla][INFO]: Initializing CUDA extension...
    2017-06-26 23:10:06,571 [nnabla][INFO]: Initializing cuDNN extension...


Setup solver for training

.. code-block:: python2

    solver = S.Adam(alpha=1e-3)
    solver.set_parameters(nn.get_parameters())

Create data iterator

.. code-block:: python2

    loss = []
    def epoch_end_callback(epoch):
        global loss
        print "[", epoch, np.mean(loss), itr, "]",
        loss = []
    
    data = data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)
    data.register_epoch_end_callback(epoch_end_callback)


.. parsed-literal::

    2017-06-26 23:10:07,221 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:07,224 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 23:10:07,226 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:07,228 [nnabla][INFO]: On-memory
    2017-06-26 23:10:07,230 [nnabla][INFO]: Using DataIterator


Perform training iterations and output training loss:

.. code-block:: python2

    %%time
    for epoch in range(30):
        itr = 0
        while data.epoch == epoch:
            x.d, t.d = data.next()
            static_l.forward(clear_no_need_grad=True)
            solver.zero_grad()
            static_l.backward(clear_buffer=True)
            solver.update()
            loss.append(static_l.d.copy())
            itr += 1
    print ''


.. parsed-literal::

    [ 0 0.909297 112 ] [ 1 0.183863 111 ] [ 2 0.0723054 111 ] [ 3 0.0653021 112 ] [ 4 0.0628503 111 ] [ 5 0.0731626 111 ] [ 6 0.0319093 112 ] [ 7 0.0610926 111 ] [ 8 0.0817437 111 ] [ 9 0.0717577 112 ] [ 10 0.0241882 111 ] [ 11 0.0119452 111 ] [ 12 0.00664761 112 ] [ 13 0.00377711 111 ] [ 14 0.000605656 111 ] [ 15 0.000236613 111 ] [ 16 0.000174549 112 ] [ 17 0.000142428 111 ] [ 18 0.000126015 111 ] [ 19 0.000111144 112 ] [ 20 0.000100751 111 ] [ 21 9.03808e-05 111 ] [ 22 8.35904e-05 112 ] [ 23 7.73492e-05 111 ] [ 24 6.91389e-05 111 ] [ 25 6.74929e-05 112 ] [ 26 6.08386e-05 111 ] [ 27 5.62182e-05 111 ] [ 28 5.33428e-05 112 ] [ 29 4.94594e-05 111 ] 
    CPU times: user 14.3 s, sys: 6.78 s, total: 21.1 s
    Wall time: 21.1 s


Dynamic computation graph
-------------------------

Now, we will use a dynamic computation graph, where the neural network
is setup each time we want to do a forward/backward pass through it.
This allows us to, e.g., randomly dropout layers or to have network
architectures that depend on input data. In this example, we will use
for simplicity the same neural network structure and only dynamically
create it. For example, adding a
``if np.random.rand() > dropout_probability:`` into ``cnn()`` allows to
dropout layers.

First, we setup the solver and the data iterator for the training:

.. code-block:: python2

    nn.clear_parameters()
    solver = S.Adam(alpha=1e-3)
    solver.set_parameters(nn.get_parameters())
    
    loss = []
    def epoch_end_callback(epoch):
        global loss
        print "[", epoch, np.mean(loss), itr, "]",
        loss = []
    data = data_iterator_tiny_digits(digits, batch_size=16, shuffle=True)
    data.register_epoch_end_callback(epoch_end_callback)


.. parsed-literal::

    2017-06-26 23:10:28,449 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:28,450 [nnabla][INFO]: Using DataSourceWithMemoryCache
    2017-06-26 23:10:28,450 [nnabla][INFO]: DataSource with shuffle(True)
    2017-06-26 23:10:28,451 [nnabla][INFO]: On-memory
    2017-06-26 23:10:28,451 [nnabla][INFO]: Using DataIterator


.. code-block:: python2

    %%time
    for epoch in range(30):
        itr = 0
        while data.epoch == epoch:
            x.d, t.d = data.next()
            with nn.auto_forward():
                dynamic_y = cnn(x)
                dynamic_l = F.mean(F.softmax_cross_entropy(dynamic_y, t))
            solver.set_parameters(nn.get_parameters(), reset=False, retain_state=True) # this can be done dynamically
            solver.zero_grad()
            dynamic_l.backward(clear_buffer=True)
            solver.update()
            loss.append(dynamic_l.d.copy())
            itr += 1
    print ''


.. parsed-literal::

    [ 0 1.04669 112 ] [ 1 0.151949 111 ] [ 2 0.093581 111 ] [ 3 0.129242 112 ] [ 4 0.0452591 111 ] [ 5 0.0343987 111 ] [ 6 0.0315372 112 ] [ 7 0.0336886 111 ] [ 8 0.0194571 111 ] [ 9 0.00923094 112 ] [ 10 0.00536065 111 ] [ 11 0.000669383 111 ] [ 12 0.000294232 112 ] [ 13 0.000245866 111 ] [ 14 0.000201116 111 ] [ 15 0.000164177 111 ] [ 16 0.00014832 112 ] [ 17 0.000131479 111 ] [ 18 0.000115171 111 ] [ 19 0.000101432 112 ] [ 20 9.06228e-05 111 ] [ 21 8.7103e-05 111 ] [ 22 7.79601e-05 112 ] [ 23 7.59678e-05 111 ] [ 24 6.64341e-05 111 ] [ 25 6.22717e-05 112 ] [ 26 5.8643e-05 111 ] [ 27 5.35373e-05 111 ] [ 28 4.96717e-05 112 ] [ 29 4.65124e-05 111 ] 
    CPU times: user 23.4 s, sys: 5.35 s, total: 28.7 s
    Wall time: 28.7 s


Comparing the two processing times, we can observe that both schemes
("static" and "dynamic") takes the same execution time, i.e., although
we created the computation graph dynamically, we did not lose
performance.
